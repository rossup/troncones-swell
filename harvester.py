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

GEFS additionally carries partitioned swell, so for that source we also
compute a direction-weighted "surf height": only swell trains pointed into
the break's window count, and windsea is discarded entirely. ECMWF open data
publishes no partitions, so it keeps the combined view only.

Runs inside GitHub Actions. No secrets, no API keys.
"""

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

TZ_OFFSET_H = -6          # Guerrero local time; mirrors TZ_OFFSET_H in index.html

# GEFS forecast hours: 6-hourly through day 7, 12-hourly to day 16
GEFS_HOURS = list(range(0, 169, 6)) + list(range(180, 385, 12))

# ECMWF steps: 12-hourly to day 15. The wave-ensemble files bundle all 51
# members per step, so every step requested costs ~150 fields of download.
ECMWF_STEPS = list(range(0, 361, 12))

GEFS_BUCKET = "https://noaa-gefs-pds.s3.amazonaws.com"
GEFS_MEMBERS = ["c00"] + [f"p{i:02d}" for i in range(1, 31)]

# ----------------------------------------------------------------------
# Directional exposure
#
# Troncones works on swell out of roughly 220-250 degrees. Energy from
# outside that arc is shadowed or arrives at an angle that will not produce a
# rideable wave, so the combined significant height (HTSGW) overstates the
# forecast whenever the spectrum is pointed elsewhere -- and it always
# includes windsea, which is not surf at all.
# ----------------------------------------------------------------------

SWELL_WINDOW_CENTER_DEG = 235.0   # middle of the 220-250 working window
SWELL_WINDOW_SHARPNESS = 3.0      # weight = cos(SHARPNESS * delta)
SWELL_WINDOW_CUTOFF_DEG = 30.0    # == 90 / SHARPNESS, the first zero crossing
N_SWELL_PARTITIONS = 3            # GEFS publishes SWELL/SWPER/SWDIR 1..3

# These GRIB messages sit contiguously in the GEFS wave files, so the whole
# partition set comes down in one byte-range request per member-hour.
GEFS_PARTITION_VARS = ("WVHGT", "SWELL", "SWPER", "SWDIR")
GEFS_SCALAR_VARS = ("HTSGW", "PERPW", "DIRPW")
GEFS_WIND_VARS = ("WIND", "WDIR")

# Partitions are only fetched out to day 10. The 16-day combined cone is
# unaffected; the direction-aware product has little skill beyond this and
# each partition block costs ~6.9 MB per member-hour.
GEFS_PARTITION_MAX_HOUR = 240

# eccodes reports all three swell partitions under one shortName, so identity
# comes from order of appearance inside the block, not from the name.
PARTITION_FIELD_BY_SHORTNAME = {"shts": "hs", "mpts": "tp", "swdir": "dir"}
WINDSEA_SHORTNAME = "shww"

OUT_DIR = "docs"
HISTORY_MAX_ENTRIES = 160   # ~40 days of 2x-daily runs, two sources each

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
    """Yield each GRIB message from a byte range holding several of them.

    A GRIB2 message opens with the magic 'GRIB' and carries its own total
    length as a big-endian 8-byte integer at offset 8, so a concatenated
    range can be walked without writing it to disk.
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
    """Decode a multi-message range into (shortName, value) in file order.

    Order is the payload here: the three swell partitions share a shortName
    and can only be told apart by position.
    """
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
    """Split ordered (shortName, value) pairs into swell trains and windsea.

    Returns ([(hs, tp, dir), ...] of length N_SWELL_PARTITIONS, windsea_hs).
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
    triples = []
    for i in range(N_SWELL_PARTITIONS):
        p = parts.get(i, {})
        triples.append((p.get("hs"), p.get("tp"), p.get("dir")))
    return triples, windsea


def direction_weight(direction_deg):
    """Exposure weight for swell arriving from `direction_deg`.

    cos(3 * delta) falls from 1.0 on the window centre to 0.707 at the edges
    of the 220-250 working window, reaching zero at +/-30 degrees. The hard
    cutoff is not redundant: cos(3 * delta) climbs back to +1.0 at delta=120,
    which would otherwise hand full weight to swell out of the north.
    """
    if direction_deg is None:
        return 0.0
    try:
        d = float(direction_deg)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(d) or math.isinf(d):
        return 0.0
    delta = abs(((d - SWELL_WINDOW_CENTER_DEG + 540.0) % 360.0) - 180.0)
    if delta >= SWELL_WINDOW_CUTOFF_DEG:
        return 0.0
    return max(0.0, math.cos(math.radians(SWELL_WINDOW_SHARPNESS * delta)))


def surf_height_from_partitions(partitions):
    """Direction-weighted swell height, plus the train driving it.

    Each swell train is scaled by its own exposure weight and the survivors
    are combined by energy, since independent trains add as the square of
    height. Windsea is not passed in at all -- it is not surf.

    Returns (height, period, direction). Height is 0.0 when trains exist but
    none are inside the window (genuinely flat), and None when there is no
    partition data at all (missing).
    """
    total_energy, driver, have_any = 0.0, None, False
    for hs, tp, dr in partitions:
        if hs is None:
            continue
        have_any = True
        if hs <= 0:
            continue
        w = direction_weight(dr)
        if w <= 0:
            continue
        wh = w * hs
        total_energy += wh * wh
        if driver is None or wh > driver[0]:
            driver = (wh, tp, dr)
    if not have_any:
        return None, None, None
    if driver is None:
        return 0.0, None, None
    return math.sqrt(total_energy), driver[1], driver[2]


def circular_mean_deg(values):
    """Mean of directions in degrees (proper circular mean)."""
    vals = [v for v in values if v is not None and not math.isnan(v)]
    if not vals:
        return None
    s = sum(math.sin(math.radians(v)) for v in vals)
    c = sum(math.cos(math.radians(v)) for v in vals)
    return round(math.degrees(math.atan2(s, c)) % 360)


def matrix_stats(matrix):
    """Percentile stats across members for each time column. NaN-tolerant.

    Columns can be entirely NaN -- partitions stop at day 10 while the
    combined cone runs to day 16 -- so numpy's all-NaN warnings are muted
    rather than allowed to flood the run log.
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


def parse_idx(text):
    """Parse a GRIB .idx into [(offset, var)] in file order."""
    rows = []
    for line in text.strip().splitlines():
        parts = line.split(":")
        if len(parts) < 5:
            continue
        try:
            rows.append((int(parts[1]), parts[3]))
        except ValueError:
            continue
    return rows


def wind_hours_for(cycle, hours):
    """One step per local day, nearest 13Z.

    The dawn-wind column only ever reads a single step per day, so paying to
    download WIND/WDIR at every step wastes roughly 1.5 GB a run. The choice
    of step mirrors the `jm` search in index.html so the two agree.
    """
    by_day = {}
    for h in hours:
        t = cycle + timedelta(hours=h)
        key = (t + timedelta(hours=TZ_OFFSET_H)).strftime("%Y-%m-%d")
        dist = abs(t.hour - 13)
        if key not in by_day or dist < by_day[key][0]:
            by_day[key] = (dist, h)
    return sorted(h for _, h in by_day.values())


def gefs_fetch_one(session, cycle, member, hour, want_wind, want_partitions):
    """Fetch the GRIB byte ranges for one member+hour. Network only.

    The partition fields (WVHGT, SWELL/SWPER/SWDIR 1..3) are contiguous in
    these files, so they come down as a single range rather than twelve.
    """
    base = (f"{GEFS_BUCKET}/gefs.{cycle:%Y%m%d}/{cycle:%H}/wave/gridded/"
            f"gefs.wave.t{cycle:%H}z.{member}.global.0p25.f{hour:03d}.grib2")
    r = session.get(base + ".idx", timeout=30)
    if r.status_code != 200:
        return member, hour, None
    rows = parse_idx(r.text)
    if not rows:
        return member, hour, None

    def end_of(i):
        return rows[i + 1][0] - 1 if i + 1 < len(rows) else ""

    def fetch_range(start, end, timeout):
        rr = session.get(base, headers={"Range": f"bytes={start}-{end}"},
                         timeout=timeout)
        return rr.content if rr.status_code in (200, 206) else None

    out = {"scalars": {}, "block": None}

    wanted = set(GEFS_SCALAR_VARS)
    if want_wind:
        wanted |= set(GEFS_WIND_VARS)
    seen = set()
    for i, (off, var) in enumerate(rows):
        if var in wanted and var not in seen:
            seen.add(var)
            blob = fetch_range(off, end_of(i), 60)
            if blob is not None:
                out["scalars"][var] = blob

    if want_partitions:
        pidx = [i for i, (_, var) in enumerate(rows)
                if var in GEFS_PARTITION_VARS]
        if pidx:
            out["block"] = fetch_range(rows[min(pidx)][0], end_of(max(pidx)), 180)

    return member, hour, out


def fetch_gefs():
    session = make_session()
    cycle = gefs_find_cycle(session)
    if cycle is None:
        raise RuntimeError("No complete GEFS cycle found on AWS")

    partition_hours = {h for h in GEFS_HOURS if h <= GEFS_PARTITION_MAX_HOUR}
    wind_hours = set(wind_hours_for(cycle, GEFS_HOURS))
    print(f"GEFS cycle: {cycle:%Y-%m-%d %HZ} | {len(GEFS_MEMBERS)} members x "
          f"{len(GEFS_HOURS)} hours "
          f"({len(partition_hours)} with partitions, {len(wind_hours)} with wind)")

    raw = {}  # (member, hour) -> {"scalars": {var: bytes}, "block": bytes}
    tasks = [(m, h) for m in GEFS_MEMBERS for h in GEFS_HOURS]
    with ThreadPoolExecutor(max_workers=12) as pool:
        futures = [pool.submit(gefs_fetch_one, session, cycle, m, h,
                               h in wind_hours, h in partition_hours)
                   for m, h in tasks]
        for fut in as_completed(futures):
            member, hour, payload = fut.result()
            if payload and (payload["scalars"] or payload["block"]):
                raw[(member, hour)] = payload

    print(f"GEFS fetched {len(raw)}/{len(tasks)} member-hours")
    if len(raw) < 0.6 * len(tasks):
        raise RuntimeError("Too many missing GEFS files; aborting this source")

    # Decode serially (eccodes is not thread-safe)
    data = {v: {} for v in ("HTSGW", "PERPW", "DIRPW", "WIND", "WDIR")}
    surf_hs, surf_tp, surf_dir, windsea = {}, {}, {}, {}
    n_blocks = 0
    for (member, hour), payload in raw.items():
        for var, blob in payload["scalars"].items():
            data[var].setdefault(member, {})[hour] = decode_point_value(
                blob, POINT_LAT, POINT_LON)
        block = payload.get("block")
        if not block:
            continue
        triples, ws = partitions_from_block(
            decode_block_points(block, POINT_LAT, POINT_LON))
        hs_s, tp_s, dir_s = surf_height_from_partitions(triples)
        surf_hs.setdefault(member, {})[hour] = hs_s
        surf_tp.setdefault(member, {})[hour] = tp_s
        surf_dir.setdefault(member, {})[hour] = dir_s
        windsea.setdefault(member, {})[hour] = ws
        n_blocks += 1
    print(f"GEFS decoded {n_blocks} partition blocks")

    return assemble_source(
        name="gefs", cycle=cycle, members=GEFS_MEMBERS, steps=GEFS_HOURS,
        hs=data["HTSGW"], per=data["PERPW"], dr=data["DIRPW"],
        wind=data["WIND"], wdir=data["WDIR"],
        surf_hs=surf_hs, surf_tp=surf_tp, surf_dir=surf_dir, windsea=windsea,
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
                    wind=None, wdir=None, surf_hs=None, surf_tp=None,
                    surf_dir=None, windsea=None):
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

    # Direction-weighted swell. Only present for sources that publish
    # partitions, which today means GEFS alone.
    if surf_hs:
        surf_m = to_matrix(surf_hs)
        out_extra["surf_hs_members"] = [clean_list(row) for row in surf_m]
        out_extra["surf_hs_stats"] = {
            k: clean_list(v) for k, v in matrix_stats(surf_m).items()}
        if surf_tp:
            surf_tp_stats = matrix_stats(to_matrix(surf_tp))
            out_extra["surf_tp_median"] = clean_list(surf_tp_stats["median"], 1)
        if surf_dir:
            out_extra["surf_dir_median"] = [
                circular_mean_deg([surf_dir.get(mem, {}).get(hh)
                                   for mem in members])
                for hh in steps]
        out_extra["swell_window"] = {
            "center_deg": SWELL_WINDOW_CENTER_DEG,
            "sharpness": SWELL_WINDOW_SHARPNESS,
            "cutoff_deg": SWELL_WINDOW_CUTOFF_DEG,
        }
    if windsea:
        windsea_stats = matrix_stats(to_matrix(windsea))
        out_extra["windsea_median"] = clean_list(windsea_stats["median"])

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
        # The surf peak gets its own index: the biggest combined sea and the
        # best in-window swell are frequently not the same timestep.
        if src.get("surf_hs_stats"):
            surf_peak = {}
            for j, t in enumerate(src["times"]):
                v = src["surf_hs_stats"]["median"][j]
                if v is None:
                    continue
                day = t[:10]
                if day not in surf_peak or v > surf_peak[day][0]:
                    surf_peak[day] = (v, j)
            entry["daily_surf_hs"] = {d: round(v, 2)
                                      for d, (v, j) in surf_peak.items()}
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

def _selftest_direction_weight():
    """The exposure curve, including the trap in cos(3*delta)."""
    assert direction_weight(SWELL_WINDOW_CENTER_DEG) == 1.0
    edge = math.cos(math.radians(45))
    assert abs(direction_weight(220) - edge) < 1e-9
    assert abs(direction_weight(250) - edge) < 1e-9
    # First zero crossing, and everything past it.
    assert direction_weight(205) == 0.0
    assert direction_weight(265) == 0.0
    assert direction_weight(180) == 0.0
    # cos(3*delta) returns to +1.0 at delta=120; without the cutoff these two
    # northern directions would score full weight at Troncones.
    assert direction_weight(355) == 0.0
    assert direction_weight(115) == 0.0
    # Weight falls off monotonically across the live part of the window.
    for a, b in zip(range(235, 265), range(236, 266)):
        assert direction_weight(a) >= direction_weight(b)
    assert direction_weight(None) == 0.0
    assert direction_weight(float("nan")) == 0.0


def _selftest_surf_height():
    """Partition combination, including the energy bound against combined sea."""
    # Two trains, one squarely in the window and one well outside it.
    trains = [(1.0, 15.0, 235.0), (1.0, 8.0, 150.0), (None, None, None)]
    h, tp, dr = surf_height_from_partitions(trains)
    assert abs(h - 1.0) < 1e-9, h          # only the in-window train survives
    assert tp == 15.0 and dr == 235.0      # driver is the in-window train
    combined = math.sqrt(sum(t[0] ** 2 for t in trains if t[0] is not None))
    assert h <= combined + 1e-9

    # Everything outside the window: genuinely flat, not missing.
    h2, tp2, dr2 = surf_height_from_partitions([(2.0, 18.0, 150.0)])
    assert h2 == 0.0 and tp2 is None and dr2 is None

    # No partition data at all: missing, not flat.
    h3, _, _ = surf_height_from_partitions([(None, None, None)])
    assert h3 is None

    # The driver is the largest *weighted* train, not the largest raw one.
    h4, tp4, dr4 = surf_height_from_partitions(
        [(1.2, 9.0, 262.0), (1.0, 16.0, 235.0)])
    assert dr4 == 235.0 and tp4 == 16.0

    # Weighted energy can never exceed the combined sea it was drawn from.
    rng = np.random.default_rng(11)
    for _ in range(500):
        t = [(float(rng.uniform(0, 4)), float(rng.uniform(4, 22)),
              float(rng.uniform(0, 360))) for _ in range(N_SWELL_PARTITIONS)]
        hh, _, _ = surf_height_from_partitions(t)
        assert hh <= math.sqrt(sum(x[0] ** 2 for x in t)) + 1e-9


def _selftest_grib_splitter():
    """The block splitter must walk messages by their declared length."""
    def msg(body, n):
        return b"GRIB" + b"\x00" * 4 + (16 + n).to_bytes(8, "big") + body
    a, b = msg(b"A" * 10, 10), msg(b"B" * 20, 20)
    got = list(split_grib_messages(a + b))
    assert got == [a, b], len(got)
    assert list(split_grib_messages(b"")) == []
    # A truncated trailing message is dropped rather than yielded short.
    assert list(split_grib_messages(a + b[:8])) == [a]


def _selftest_partition_mapping():
    """Partitions are identified by order, since they share a shortName."""
    decoded = [("shww", 0.38),
               ("shts", 0.98), ("shts", 0.38), ("shts", 0.28),
               ("mpww", 4.71),
               ("mpts", 15.65), ("mpts", 7.21), ("mpts", 18.0),
               ("wvdir", 280.5),
               ("swdir", 214.77), ("swdir", 157.47), ("swdir", 186.06)]
    triples, windsea = partitions_from_block(decoded)
    assert windsea == 0.38
    assert triples[0] == (0.98, 15.65, 214.77)
    assert triples[1] == (0.38, 7.21, 157.47)
    assert triples[2] == (0.28, 18.0, 186.06)
    # This is the real f024 sample: only train 1 is inside the window, so the
    # surf height must come in far below the ~1.15 m combined sea.
    h, tp, dr = surf_height_from_partitions(triples)
    assert 0.45 < h < 0.50, h
    assert dr == 214.77 and tp == 15.65


def selftest():
    global OUT_DIR
    _selftest_direction_weight()
    _selftest_surf_height()
    _selftest_grib_splitter()
    _selftest_partition_mapping()

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
    # Surf fields only cover the first half of the window, mirroring the way
    # real partitions stop at day 10 while the combined cone runs to day 16.
    cut = steps[:len(steps) // 2]
    shs, stp, sdr, wsea = {}, {}, {}, {}
    for m in members:
        for h in cut:
            trains = [(hs[m][h], per[m][h], dr[m][h]),
                      (0.4, 7.0, 150.0)]
            a, b, c = surf_height_from_partitions(trains)
            shs.setdefault(m, {})[h] = a
            stp.setdefault(m, {})[h] = b
            sdr.setdefault(m, {})[h] = c
            wsea.setdefault(m, {})[h] = 0.4

    src = assemble_source("test", cycle, members, steps, hs, per, dr, "test",
                          wind=wnd, wdir=wdd, surf_hs=shs, surf_tp=stp,
                          surf_dir=sdr, windsea=wsea)
    assert len(src["wind_median"]) == len(steps)
    assert all(0 <= d < 360 for d in src["wdir_median"])
    assert len(src["times"]) == len(steps)
    assert len(src["hs_members"]) == 31
    med = src["hs_stats"]["median"]
    assert all(src["hs_stats"]["p25"][j] <= med[j] <= src["hs_stats"]["p75"][j]
               for j in range(len(steps)))
    assert all(0 <= d < 360 for d in src["dir_median"])

    # Surf arrays are full length, populated where partitions existed and
    # None beyond the cut, so the page can render a short surf series against
    # a long combined series.
    surf_med = src["surf_hs_stats"]["median"]
    assert len(surf_med) == len(steps)
    assert all(v is not None for v in surf_med[:len(cut)])
    assert all(v is None for v in surf_med[len(cut):])
    assert len(src["surf_hs_members"]) == 31
    assert len(src["windsea_median"]) == len(steps)
    assert src["swell_window"]["center_deg"] == SWELL_WINDOW_CENTER_DEG
    # The weighted height must sit at or below the combined sea everywhere.
    for j in range(len(cut)):
        assert surf_med[j] <= med[j] + 1e-9, (j, surf_med[j], med[j])

    OUT_DIR = tempfile.mkdtemp(prefix="troncones-selftest-")
    with open(os.path.join(OUT_DIR, "forecast.json"), "w") as f:
        json.dump({"sources": {"test": src}}, f)
    update_history([src])
    with open(os.path.join(OUT_DIR, "history.json")) as f:
        hist = json.load(f)
    assert hist and "daily_surf_hs" in hist[-1]
    print(f"selftest passed (scratch output in {OUT_DIR})")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        main()
