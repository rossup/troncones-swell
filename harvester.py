"""
Troncones swell ensemble harvester.

Pulls wave forecasts from two public ensembles:
  - NOAA GEFS-Wave  (31 members, out to 16 days)  -- public domain
  - ECMWF ENS open data (51 members, out to 15 days) -- CC-BY-4.0

For one deep-water point offshore of Manzanillo Bay, it extracts
significant wave height, period, and direction from every member,
computes ensemble statistics, and writes:

  docs/forecast.json  -- the latest full forecast package
  docs/history.json   -- a rolling archive of past runs (survival tracker)

Runs inside GitHub Actions. No secrets, no API keys.
"""

import io
import json
import math
import os
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone

import numpy as np
import requests
from requests.adapters import HTTPAdapter, Retry

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

POINT_LAT = 17.5          # deep water offshore of Manzanillo Bay / Troncones
POINT_LON = -102.0        # negative = west
POINT_LABEL = "Offshore Troncones (17.5N 102.0W)"

# GEFS forecast hours: 6-hourly through day 7, 12-hourly to day 16
GEFS_HOURS = list(range(0, 169, 6)) + list(range(180, 385, 12))

# ECMWF steps: 12-hourly to day 15. The wave-ensemble files bundle all 51
# members per step, so every step requested costs ~150 fields of download.
ECMWF_STEPS = list(range(0, 361, 12))

GEFS_BUCKET = "https://noaa-gefs-pds.s3.amazonaws.com"
GEFS_MEMBERS = ["c00"] + [f"p{i:02d}" for i in range(1, 31)]

OUT_DIR = "docs"
HISTORY_MAX_ENTRIES = 160   # ~40 days of 2x-daily runs, two sources each

# FAST=1: quick test run (~2-3 min). 6 GEFS members, days 0-3, GEFS only,
# written to test_out/ so the live page in docs/ is never touched.
# Add FAST_ECMWF=1 to also pull ECMWF for days 0-3 (adds a few minutes).
FAST = os.environ.get("FAST") == "1"
FAST_ECMWF = os.environ.get("FAST_ECMWF") == "1"
if FAST:
    GEFS_MEMBERS = GEFS_MEMBERS[:6]
    GEFS_HOURS = list(range(0, 73, 6))
    ECMWF_STEPS = list(range(0, 73, 12))
    OUT_DIR = "test_out"
    print(f"FAST mode: {len(GEFS_MEMBERS)} GEFS members x {len(GEFS_HOURS)} "
          f"hours, ECMWF {'on' if FAST_ECMWF else 'off'}, output -> {OUT_DIR}/")

# ----------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------

def make_session():
    s = requests.Session()
    retry = Retry(total=3, backoff_factor=0.5,
                  status_forcelist=[500, 502, 503, 504])
    s.mount("https://", HTTPAdapter(max_retries=retry, pool_maxsize=20))
    s.headers["User-Agent"] = "troncones-swell (personal surf forecast)"
    return s


_GRID_CACHE = {}

def decode_point_value(message_bytes, lat, lon):
    """Decode one GRIB message from memory, return value at nearest wet point.

    The nearest-point search is expensive, so we do it once per grid geometry
    and cache the flat indexes of the four nearest points; every later message
    on the same grid is a single direct element read.
    """
    import eccodes
    gid = eccodes.codes_new_from_message(message_bytes)
    try:
        try:
            key = tuple(eccodes.codes_get(gid, k) for k in
                        ("Ni", "Nj", "latitudeOfFirstGridPointInDegrees",
                         "longitudeOfFirstGridPointInDegrees",
                         "iDirectionIncrementInDegrees", "jScansPositively"))
        except Exception:
            key = None

        if key is not None and key in _GRID_CACHE:
            for idx in _GRID_CACHE[key]:
                try:
                    v = eccodes.codes_get_double_element(gid, "values", idx)
                except Exception:
                    continue
                if v is not None and abs(v) < 9000:
                    return float(v)
            return None

        for lon_try in (lon, lon % 360):
            try:
                nearest = eccodes.codes_grib_find_nearest(gid, lat, lon_try,
                                                          npoints=4)
            except Exception:
                continue
            if key is not None:
                try:
                    _GRID_CACHE[key] = [int(pt.index) for pt in nearest]
                except Exception:
                    pass
            for pt in nearest:
                v = pt.value
                if v is not None and abs(v) < 9000:
                    return float(v)
        return None
    finally:
        eccodes.codes_release(gid)


def circular_mean_deg(values):
    """Mean of directions in degrees (proper circular mean)."""
    vals = [v for v in values if v is not None and not math.isnan(v)]
    if not vals:
        return None
    s = sum(math.sin(math.radians(v)) for v in vals)
    c = sum(math.cos(math.radians(v)) for v in vals)
    return round(math.degrees(math.atan2(s, c)) % 360)


def matrix_stats(matrix):
    """Percentile stats across members for each time column. NaN-tolerant."""
    with np.errstate(all="ignore"):
        return {
            "median": np.nanmedian(matrix, axis=0),
            "p25": np.nanpercentile(matrix, 25, axis=0),
            "p75": np.nanpercentile(matrix, 75, axis=0),
            "p10": np.nanpercentile(matrix, 10, axis=0),
            "p90": np.nanpercentile(matrix, 90, axis=0),
        }


def clean(x, nd=2):
    """Round for JSON; NaN -> None."""
    if x is None:
        return None
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(xf) or math.isinf(xf):
        return None
    return round(xf, nd)


def clean_list(seq, nd=2):
    return [clean(v, nd) for v in seq]


# ----------------------------------------------------------------------
# Source 1: NOAA GEFS-Wave via AWS Open Data (byte-range subsetting)
# ----------------------------------------------------------------------

def gefs_find_cycle(session):
    """Newest cycle whose files are fully published (check last file, last member)."""
    now = datetime.now(timezone.utc)
    for hours_back in range(6, 6 * 9, 6):
        t = now - timedelta(hours=hours_back)
        cycle = t.replace(hour=(t.hour // 6) * 6, minute=0, second=0, microsecond=0)
        url = (f"{GEFS_BUCKET}/gefs.{cycle:%Y%m%d}/{cycle:%H}/wave/gridded/"
               f"gefs.wave.t{cycle:%H}z.p30.global.0p25.f384.grib2.idx")
        try:
            r = session.head(url, timeout=20)
            if r.status_code == 200:
                return cycle
        except requests.RequestException:
            pass
    return None


def gefs_fetch_one(session, cycle, member, hour):
    """Fetch HTSGW/PERPW/DIRPW GRIB messages for one member+hour. Network only."""
    base = (f"{GEFS_BUCKET}/gefs.{cycle:%Y%m%d}/{cycle:%H}/wave/gridded/"
            f"gefs.wave.t{cycle:%H}z.{member}.global.0p25.f{hour:03d}.grib2")
    r = session.get(base + ".idx", timeout=30)
    if r.status_code != 200:
        return member, hour, None
    lines = r.text.strip().splitlines()
    offsets, wanted = [], {}
    for i, line in enumerate(lines):
        parts = line.split(":")
        if len(parts) < 5:
            continue
        offsets.append(int(parts[1]))
        var = parts[3]
        if var in ("HTSGW", "PERPW", "DIRPW", "WIND", "WDIR"):
            wanted[var] = i
    out = {}
    for var, i in wanted.items():
        start = offsets[i]
        end = offsets[i + 1] - 1 if i + 1 < len(offsets) else ""
        rr = session.get(base, headers={"Range": f"bytes={start}-{end}"}, timeout=60)
        if rr.status_code in (200, 206):
            out[var] = rr.content
    return member, hour, out


def fetch_gefs():
    session = make_session()
    cycle = gefs_find_cycle(session)
    if cycle is None:
        raise RuntimeError("No complete GEFS cycle found on AWS")
    print(f"GEFS cycle: {cycle:%Y-%m-%d %HZ} | {len(GEFS_MEMBERS)} members x "
          f"{len(GEFS_HOURS)} hours")

    raw = {}  # (member, hour) -> {var: bytes}
    tasks = [(m, h) for m in GEFS_MEMBERS for h in GEFS_HOURS]
    with ThreadPoolExecutor(max_workers=12) as pool:
        futures = [pool.submit(gefs_fetch_one, session, cycle, m, h)
                   for m, h in tasks]
        for fut in as_completed(futures):
            member, hour, blobs = fut.result()
            if blobs:
                raw[(member, hour)] = blobs

    print(f"GEFS fetched {len(raw)}/{len(tasks)} member-hours")
    if len(raw) < 0.6 * len(tasks):
        raise RuntimeError("Too many missing GEFS files; aborting this source")

    # Decode serially (eccodes is not thread-safe)
    data = {v: {} for v in ("HTSGW", "PERPW", "DIRPW", "WIND", "WDIR")}
    for (member, hour), blobs in raw.items():
        for var, blob in blobs.items():
            val = decode_point_value(blob, POINT_LAT, POINT_LON)
            data[var].setdefault(member, {})[hour] = val

    return assemble_source(
        name="gefs", cycle=cycle, members=GEFS_MEMBERS, steps=GEFS_HOURS,
        hs=data["HTSGW"], per=data["PERPW"], dr=data["DIRPW"],
        wind=data["WIND"], wdir=data["WDIR"],
        period_note="peak period of primary swell (PERPW)",
    )


# ----------------------------------------------------------------------
# Source 2: ECMWF ENS via official open-data client
# ----------------------------------------------------------------------

def fetch_ecmwf():
    import eccodes
    from ecmwf.opendata import Client

    client = Client(source="ecmwf")
    # Wave ensemble lives in its own stream: waef, type "ef" (all 50+1 members
    # bundled per step). The atmospheric ensemble stream (enfo) has no wave fields.
    request = dict(stream="waef", type="ef",
                   param=["swh", "mwp", "mwd"], step=ECMWF_STEPS)

    # Candidate cycles, newest first. If the newest is still mid-publication
    # (files appear progressively over ~2h), fall back one cycle at a time.
    candidates = []
    try:
        latest = client.latest(**request)
        print(f"ECMWF latest ENS cycle: {latest:%Y-%m-%d %HZ}")
        candidates = [latest - timedelta(hours=12 * i) for i in range(3)]
    except Exception as e:
        print(f"ECMWF latest() unavailable ({e}); trying recent cycles blind")
        now = datetime.now(timezone.utc)
        base = now.replace(hour=(0 if now.hour < 12 else 12), minute=0,
                           second=0, microsecond=0)
        candidates = [base - timedelta(hours=12 * i) for i in range(1, 4)]

    tmp = tempfile.NamedTemporaryFile(suffix=".grib2", delete=False)
    tmp.close()
    last_err = None
    for cyc in candidates:
        req = dict(request, date=cyc.strftime("%Y%m%d"), time=cyc.hour)
        try:
            client.retrieve(target=tmp.name, **req)
            print(f"ECMWF using cycle {cyc:%Y-%m-%d %HZ}")
            last_err = None
            break
        except Exception as e:
            last_err = e
            print(f"ECMWF cycle {cyc:%Y-%m-%d %HZ} not complete yet ({e}); "
                  f"stepping back")
    if last_err is not None:
        raise RuntimeError(f"No complete ECMWF cycle available: {last_err}")
    size_mb = os.path.getsize(tmp.name) / 1e6
    print(f"ECMWF download complete: {size_mb:.0f} MB")

    data = {v: {} for v in ("swh", "mwp", "mwd")}
    cycle = None
    n_msgs = 0
    with open(tmp.name, "rb") as f:
        while True:
            gid = eccodes.codes_grib_new_from_file(f)
            if gid is None:
                break
            n_msgs += 1
            if n_msgs % 1000 == 0:
                print(f"ECMWF decoding... {n_msgs} fields")
            try:
                short = eccodes.codes_get(gid, "shortName")
                if short not in data:
                    continue
                step = int(eccodes.codes_get(gid, "endStep"))
                try:
                    number = int(eccodes.codes_get(gid, "number"))
                except Exception:
                    number = 0
                if cycle is None:
                    d = str(eccodes.codes_get(gid, "dataDate"))
                    t = int(eccodes.codes_get(gid, "dataTime")) // 100
                    cycle = datetime.strptime(d, "%Y%m%d").replace(
                        hour=t, tzinfo=timezone.utc)
                msg = eccodes.codes_get_message(gid)
                val = decode_point_value(msg, POINT_LAT, POINT_LON)
                data[short].setdefault(number, {})[step] = val
            finally:
                eccodes.codes_release(gid)
    os.unlink(tmp.name)

    members = sorted(data["swh"].keys())
    print(f"ECMWF decoded {len(members)} members")
    if len(members) < 20:
        raise RuntimeError("Too few ECMWF members decoded")

    return assemble_source(
        name="ecmwf", cycle=cycle, members=members, steps=ECMWF_STEPS,
        hs=data["swh"], per=data["mwp"], dr=data["mwd"],
        period_note="mean wave period (mwp); reads ~15-20% lower than peak period",
    )


# ----------------------------------------------------------------------
# Shared assembly + statistics
# ----------------------------------------------------------------------

def assemble_source(name, cycle, members, steps, hs, per, dr, period_note,
                    wind=None, wdir=None):
    times = [(cycle + timedelta(hours=h)).strftime("%Y-%m-%dT%H:%MZ")
             for h in steps]

    def to_matrix(d):
        m = np.full((len(members), len(steps)), np.nan)
        for i, mem in enumerate(members):
            row = d.get(mem, {})
            for j, h in enumerate(steps):
                v = row.get(h)
                if v is not None:
                    m[i, j] = v
        return m

    hs_m, per_m = to_matrix(hs), to_matrix(per)
    hs_stats = matrix_stats(hs_m)
    per_stats = matrix_stats(per_m)

    dir_median = []
    for j, h in enumerate(steps):
        col = [dr.get(mem, {}).get(h) for mem in members]
        dir_median.append(circular_mean_deg(col))

    out_extra = {}
    if wind:
        wind_m = to_matrix(wind)
        wind_stats = matrix_stats(wind_m)
        out_extra["wind_members"] = [clean_list(row, 1) for row in wind_m]
        out_extra["wind_median"] = clean_list(wind_stats["median"], 1)
        out_extra["wdir_median"] = [
            circular_mean_deg([wdir.get(mem, {}).get(hh) for mem in members])
            for hh in steps] if wdir else None

    return {
        **out_extra,
        "name": name,
        "cycle_utc": cycle.strftime("%Y-%m-%dT%H:%MZ"),
        "n_members": len(members),
        "period_note": period_note,
        "times": times,
        "hs_members": [clean_list(row) for row in hs_m],
        "tp_members": [clean_list(row, 1) for row in per_m],
        "hs_stats": {k: clean_list(v) for k, v in hs_stats.items()},
        "tp_median": clean_list(per_stats["median"], 1),
        "dir_median": dir_median,
    }


def update_history(sources):
    path = os.path.join(OUT_DIR, "history.json")
    history = []
    if os.path.exists(path):
        try:
            with open(path) as f:
                history = json.load(f)
        except Exception:
            history = []

    run_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
    for src in sources:
        peak = {}
        for j, t in enumerate(src["times"]):
            v = src["hs_stats"]["median"][j]
            if v is None:
                continue
            day = t[:10]
            if day not in peak or v > peak[day][0]:
                peak[day] = (v, j)
        entry = {
            "run_utc": run_utc,
            "source": src["name"],
            "cycle_utc": src["cycle_utc"],
            "daily_median_hs": {d: round(v, 2) for d, (v, j) in peak.items()},
            "daily_tp": {d: src["tp_median"][j] for d, (v, j) in peak.items()},
            "daily_dir": {d: src["dir_median"][j] for d, (v, j) in peak.items()},
        }
        if src.get("wind_median"):
            entry["daily_wind"] = {d: src["wind_median"][j]
                                   for d, (v, j) in peak.items()}
        history.append(entry)

    history = history[-HISTORY_MAX_ENTRIES:]
    with open(path, "w") as f:
        json.dump(history, f, separators=(",", ":"))
    print(f"History: {len(history)} entries")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()
    sources, errors = [], []

    for label, fn in (("GEFS", fetch_gefs), ("ECMWF", fetch_ecmwf)):
        if FAST and label == "ECMWF" and not FAST_ECMWF:
            print("ECMWF: skipped (FAST mode)")
            continue
        try:
            sources.append(fn())
            print(f"{label}: OK ({time.time()-t0:.0f}s elapsed)")
        except Exception as e:
            errors.append(f"{label}: {e}")
            print(f"{label} FAILED: {e}")

    if not sources:
        print("Both sources failed:", errors)
        sys.exit(1)

    forecast = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
        "point": {"lat": POINT_LAT, "lon": POINT_LON, "label": POINT_LABEL},
        "sources": {s["name"]: s for s in sources},
        "errors": errors,
        "attribution": ("NOAA GEFS-Wave (public domain); ECMWF open data "
                        "(CC-BY-4.0, modified)"),
    }
    with open(os.path.join(OUT_DIR, "forecast.json"), "w") as f:
        json.dump(forecast, f, separators=(",", ":"))

    update_history(sources)
    print(f"Done in {time.time()-t0:.0f}s. Sources: "
          f"{[s['name'] for s in sources]}")


# ----------------------------------------------------------------------
# Self-test with synthetic data (no network) -- used during development
# ----------------------------------------------------------------------

def selftest():
    rng = np.random.default_rng(7)
    members = [f"m{i}" for i in range(31)]
    steps = list(range(0, 73, 6))
    cycle = datetime(2026, 8, 20, 0, tzinfo=timezone.utc)
    base = 1.2 + 0.6 * np.sin(np.linspace(0, 3, len(steps)))
    hs = {m: {h: float(base[j] + rng.normal(0, 0.15 + 0.02 * j))
              for j, h in enumerate(steps)} for m in members}
    per = {m: {h: float(12 + rng.normal(0, 1)) for h in steps} for m in members}
    dr = {m: {h: float((200 + rng.normal(0, 8)) % 360) for h in steps}
          for m in members}
    wnd = {m: {h: float(6 + rng.normal(0, 2)) for h in steps} for m in members}
    wdd = {m: {h: float((50 + rng.normal(0, 15)) % 360) for h in steps} for m in members}
    src = assemble_source("test", cycle, members, steps, hs, per, dr, "test",
                          wind=wnd, wdir=wdd)
    assert len(src["wind_median"]) == len(steps)
    assert all(0 <= d < 360 for d in src["wdir_median"])
    assert len(src["times"]) == len(steps)
    assert len(src["hs_members"]) == 31
    med = src["hs_stats"]["median"]
    assert all(src["hs_stats"]["p25"][j] <= med[j] <= src["hs_stats"]["p75"][j]
               for j in range(len(steps)))
    assert all(0 <= d < 360 for d in src["dir_median"])
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "forecast.json"), "w") as f:
        json.dump({"sources": {"test": src}}, f)
    update_history([src])
    print("selftest passed")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        main()
