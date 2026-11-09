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
import warnings
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

# Swell partitions (GEFS only; ECMWF open data has none).
# WVHGT + SWELL/SWPER/SWDIR 1..3 (+ the windsea period/direction between
# them) sit contiguously in each GEFS wave file, so the whole set comes down
# as one byte-range request per member-hour. Partitions are fetched to day 10
# only: each block is several MB and skill beyond that is low.
N_SWELL_PARTITIONS = 3
GEFS_PARTITION_VARS = ("WVHGT", "SWELL", "SWPER", "SWDIR")
GEFS_PARTITION_MAX_HOUR = 240
# eccodes shortNames inside the block. The three swell partitions share a
# shortName, so which partition is which comes from order of appearance.
PARTITION_FIELD_BY_SHORTNAME = {"shts": "hs", "mpts": "tp", "swdir": "dir"}
WINDSEA_SHORTNAME = "shww"

OUT_DIR = "docs"
HISTORY_MAX_ENTRIES = 160   # ~40 days of 2x-daily runs, two sources each

# FAST=1: quick test run (~2-3 min). 6 GEFS members, days 0-3, GEFS only,
# written to test_out/ so the live page in docs/ is never touched.
FAST = os.environ.get("FAST") == "1"
# ECMWF is off by default: its open data has no swell partitions, only a
# mean direction over all waves. ECMWF=1 switches it back on (full or FAST).
USE_ECMWF = os.environ.get("ECMWF") == "1"
if FAST:
    GEFS_MEMBERS = GEFS_MEMBERS[:6]
    GEFS_HOURS = list(range(0, 73, 6))
    ECMWF_STEPS = list(range(0, 73, 12))
    OUT_DIR = "test_out"
    print(f"FAST mode: {len(GEFS_MEMBERS)} GEFS members x {len(GEFS_HOURS)} "
          f"hours, ECMWF {'on' if USE_ECMWF else 'off'}, output -> {OUT_DIR}/")

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

def _point_value_from_gid(gid, lat, lon):
    """Value at the nearest wet point for an already-opened GRIB handle.

    The nearest-point search is expensive, so we do it once per grid geometry
    and cache the flat indexes of the four nearest points; every later message
    on the same grid is a single direct element read.
    """
    import eccodes
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


def decode_point_value(message_bytes, lat, lon):
    """Decode one GRIB message from memory, return value at nearest wet point."""
    import eccodes
    gid = eccodes.codes_new_from_message(message_bytes)
    try:
        return _point_value_from_gid(gid, lat, lon)
    finally:
        eccodes.codes_release(gid)


def split_grib_messages(blob):
    """Yield each GRIB2 message from a byte range holding several of them.

    A GRIB2 message opens with b'GRIB' and carries its total length as a
    big-endian 8-byte integer at offset 8.
    """
    i, n = 0, len(blob)
    while i + 16 <= n:
        if blob[i:i + 4] != b"GRIB":
            i += 1
            continue
        length = int.from_bytes(blob[i + 8:i + 16], "big")
        if length <= 0 or i + length > n:
            return
        yield blob[i:i + length]
        i += length


def decode_block_points(blob, lat, lon):
    """Decode a multi-message range into [(shortName, value)] in file order."""
    import eccodes
    out = []
    for msg in split_grib_messages(blob):
        try:
            gid = eccodes.codes_new_from_message(msg)
        except Exception:
            continue
        try:
            try:
                short = eccodes.codes_get(gid, "shortName")
            except Exception:
                continue
            out.append((short, _point_value_from_gid(gid, lat, lon)))
        finally:
            eccodes.codes_release(gid)
    return out


def partitions_from_block(decoded):
    """Split ordered (shortName, value) pairs into swell partitions + windsea.

    Returns ([(hs, tp, dir)] * N_SWELL_PARTITIONS, windsea_hs).
    """
    seen, parts, windsea = {}, {}, None
    for short, val in decoded:
        if short == WINDSEA_SHORTNAME:
            if windsea is None:
                windsea = val
            continue
        field = PARTITION_FIELD_BY_SHORTNAME.get(short)
        if field is None:
            continue
        i = seen.get(short, 0)
        seen[short] = i + 1
        if i < N_SWELL_PARTITIONS:
            parts.setdefault(i, {})[field] = val
    triples = [(parts.get(i, {}).get("hs"), parts.get(i, {}).get("tp"),
                parts.get(i, {}).get("dir")) for i in range(N_SWELL_PARTITIONS)]
    return triples, windsea


def circular_mean_deg(values):
    """Mean of directions in degrees (proper circular mean)."""
    vals = [v for v in values if v is not None and not math.isnan(v)]
    if not vals:
        return None
    s = sum(math.sin(math.radians(v)) for v in vals)
    c = sum(math.cos(math.radians(v)) for v in vals)
    return round(math.degrees(math.atan2(s, c)) % 360)


def circular_spread_deg(values):
    """Circular standard deviation of directions, in degrees.

    sqrt(-2 ln R), where R is the length of the mean unit vector. 0 when all
    members agree; ~40+ when they are scattered across a quadrant or more.
    """
    vals = [v for v in values if v is not None and not math.isnan(v)]
    if len(vals) < 2:
        return None
    s = sum(math.sin(math.radians(v)) for v in vals) / len(vals)
    c = sum(math.cos(math.radians(v)) for v in vals) / len(vals)
    r = min(1.0, math.hypot(s, c))
    if r <= 1e-12:
        return 180
    return round(math.degrees(math.sqrt(-2.0 * math.log(r))))


def matrix_stats(matrix):
    """Percentile stats across members for each time column. NaN-tolerant.

    Columns can be entirely NaN (swell partitions stop at day 10), so numpy's
    all-NaN warnings are muted rather than flooding the log.
    """
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
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


def gefs_fetch_one(session, cycle, member, hour, want_partitions=False):
    """Fetch HTSGW/PERPW/DIRPW/WIND/WDIR GRIB messages for one member+hour,
    plus (if asked) the contiguous swell-partition block under "__block__".
    Network only."""
    base = (f"{GEFS_BUCKET}/gefs.{cycle:%Y%m%d}/{cycle:%H}/wave/gridded/"
            f"gefs.wave.t{cycle:%H}z.{member}.global.0p25.f{hour:03d}.grib2")
    r = session.get(base + ".idx", timeout=30)
    if r.status_code != 200:
        return member, hour, None
    rows = []  # (offset, var) in file order
    for line in r.text.strip().splitlines():
        parts = line.split(":")
        if len(parts) < 5:
            continue
        try:
            rows.append((int(parts[1]), parts[3]))
        except ValueError:
            continue

    def end_of(i):
        return rows[i + 1][0] - 1 if i + 1 < len(rows) else ""

    def fetch_range(start, end, timeout):
        rr = session.get(base, headers={"Range": f"bytes={start}-{end}"},
                         timeout=timeout)
        return rr.content if rr.status_code in (200, 206) else None

    wanted = {}
    for i, (_, var) in enumerate(rows):
        if var in ("HTSGW", "PERPW", "DIRPW", "WIND", "WDIR"):
            wanted[var] = i
    out = {}
    for var, i in wanted.items():
        blob = fetch_range(rows[i][0], end_of(i), 60)
        if blob is not None:
            out[var] = blob

    if want_partitions:
        pidx = [i for i, (_, var) in enumerate(rows) if var in GEFS_PARTITION_VARS]
        if pidx:
            blob = fetch_range(rows[min(pidx)][0], end_of(max(pidx)), 180)
            if blob is not None:
                out["__block__"] = blob
    return member, hour, out


def fetch_gefs():
    session = make_session()
    cycle = gefs_find_cycle(session)
    if cycle is None:
        raise RuntimeError("No complete GEFS cycle found on AWS")
    partition_hours = {h for h in GEFS_HOURS if h <= GEFS_PARTITION_MAX_HOUR}
    print(f"GEFS cycle: {cycle:%Y-%m-%d %HZ} | {len(GEFS_MEMBERS)} members x "
          f"{len(GEFS_HOURS)} hours ({len(partition_hours)} with swell partitions)")

    raw = {}  # (member, hour) -> {var: bytes}
    tasks = [(m, h) for m in GEFS_MEMBERS for h in GEFS_HOURS]
    with ThreadPoolExecutor(max_workers=12) as pool:
        futures = [pool.submit(gefs_fetch_one, session, cycle, m, h,
                               h in partition_hours)
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
    # swell[n][field][member][hour], n = 0..N-1, field in hs/tp/dir
    swell = [{"hs": {}, "tp": {}, "dir": {}} for _ in range(N_SWELL_PARTITIONS)]
    windsea = {}
    n_blocks = 0
    for (member, hour), blobs in raw.items():
        for var, blob in blobs.items():
            if var == "__block__":
                continue
            val = decode_point_value(blob, POINT_LAT, POINT_LON)
            data[var].setdefault(member, {})[hour] = val
        block = blobs.get("__block__")
        if block:
            triples, ws = partitions_from_block(
                decode_block_points(block, POINT_LAT, POINT_LON))
            for n, (hs_, tp_, dir_) in enumerate(triples):
                swell[n]["hs"].setdefault(member, {})[hour] = hs_
                swell[n]["tp"].setdefault(member, {})[hour] = tp_
                swell[n]["dir"].setdefault(member, {})[hour] = dir_
            windsea.setdefault(member, {})[hour] = ws
            n_blocks += 1
    print(f"GEFS decoded {n_blocks} swell-partition blocks")

    return assemble_source(
        name="gefs", cycle=cycle, members=GEFS_MEMBERS, steps=GEFS_HOURS,
        hs=data["HTSGW"], per=data["PERPW"], dr=data["DIRPW"],
        wind=data["WIND"], wdir=data["WDIR"],
        swell=swell if n_blocks else None, windsea=windsea if n_blocks else None,
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
                    wind=None, wdir=None, swell=None, windsea=None):
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

    # Swell partitions (GEFS only). Partition 1 is the most energetic swell
    # train in each member, 2 the next, etc. Arrays are full length; steps
    # past GEFS_PARTITION_MAX_HOUR are None.
    #   swN_hs_median / swN_tp_median : ensemble median height (m) / period (s)
    #   swN_dir_mean                  : circular mean direction (deg, "from")
    #   swN_dir_spread                : circular std dev across members (deg)
    if swell:
        for n, part in enumerate(swell, start=1):
            out_extra[f"sw{n}_hs_median"] = clean_list(
                matrix_stats(to_matrix(part["hs"]))["median"])
            out_extra[f"sw{n}_tp_median"] = clean_list(
                matrix_stats(to_matrix(part["tp"]))["median"], 1)
            cols = [[part["dir"].get(mem, {}).get(h) for mem in members]
                    for h in steps]
            out_extra[f"sw{n}_dir_mean"] = [circular_mean_deg(c) for c in cols]
            out_extra[f"sw{n}_dir_spread"] = [circular_spread_deg(c) for c in cols]
    if windsea:
        out_extra["windsea_hs_median"] = clean_list(
            matrix_stats(to_matrix(windsea))["median"])

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
        if label == "ECMWF" and not USE_ECMWF:
            print("ECMWF: skipped (set ECMWF=1 to include)")
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

def _selftest_partitions():
    # GRIB splitter walks messages by declared length; drops a truncated tail.
    def msg(body):
        return b"GRIB" + b"\x00" * 4 + (16 + len(body)).to_bytes(8, "big") + body
    a, b = msg(b"A" * 10), msg(b"B" * 20)
    assert list(split_grib_messages(a + b)) == [a, b]
    assert list(split_grib_messages(a + b[:8])) == [a]
    # Partitions share a shortName, so identity comes from order.
    decoded = [("shww", 0.38),
               ("shts", 0.98), ("shts", 0.38), ("shts", 0.28),
               ("mpww", 4.71),
               ("mpts", 15.65), ("mpts", 7.21), ("mpts", 18.0),
               ("wvdir", 280.5),
               ("swdir", 214.77), ("swdir", 157.47), ("swdir", 186.06)]
    triples, ws = partitions_from_block(decoded)
    assert ws == 0.38
    assert triples == [(0.98, 15.65, 214.77), (0.38, 7.21, 157.47),
                       (0.28, 18.0, 186.06)]
    # Circular spread: agreement -> 0, wraps across north, scatter -> large.
    assert circular_spread_deg([220, 220, 220]) == 0
    assert circular_spread_deg([355, 5]) == circular_spread_deg([175, 185])
    assert 4 <= circular_spread_deg([355, 5]) <= 6
    assert circular_spread_deg([0, 90, 180, 270]) == 180
    assert circular_spread_deg([200]) is None


def selftest():
    _selftest_partitions()
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
    # Synthetic partitions on the first half of the steps only (real ones stop
    # at day 10): train 1 from ~215 deg (tight), train 2 from ~160 (looser).
    cut = steps[:len(steps) // 2]
    sw = [{"hs": {}, "tp": {}, "dir": {}} for _ in range(N_SWELL_PARTITIONS)]
    centers, spreads = (215, 160, 190), (5, 20, 30)
    for m in members:
        for h in cut:
            for n in range(N_SWELL_PARTITIONS):
                sw[n]["hs"].setdefault(m, {})[h] = 1.0 / (n + 1)
                sw[n]["tp"].setdefault(m, {})[h] = 15.0 - 3 * n
                sw[n]["dir"].setdefault(m, {})[h] = float(
                    (centers[n] + rng.normal(0, spreads[n])) % 360)
    wsea = {m: {h: 0.3 for h in cut} for m in members}
    src = assemble_source("test", cycle, members, steps, hs, per, dr, "test",
                          wind=wnd, wdir=wdd, swell=sw, windsea=wsea)
    k = len(cut)
    for n in range(1, N_SWELL_PARTITIONS + 1):
        for key in ("hs_median", "tp_median", "dir_mean", "dir_spread"):
            arr = src[f"sw{n}_{key}"]
            assert len(arr) == len(steps), (n, key)
            assert all(v is not None for v in arr[:k]), (n, key)
            assert all(v is None for v in arr[k:]), (n, key)
    assert all(abs(d - 215) <= 5 for d in src["sw1_dir_mean"][:k])
    assert all(abs(d - 160) <= 12 for d in src["sw2_dir_mean"][:k])
    assert max(src["sw1_dir_spread"][:k]) < min(src["sw2_dir_spread"][:k])
    assert len(src["windsea_hs_median"]) == len(steps)
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
