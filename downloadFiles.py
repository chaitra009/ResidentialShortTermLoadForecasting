"""
downloadFiles.py — FIXED VERSION
==================================
Changes from original:
  1. GISJOIN list is now derived dynamically from selected_buildings.csv
     instead of being hardcoded. This means re-running extractColumns.py
     with different parameters (different random_state, different zone
     quotas, etc.) will automatically stay in sync — no more silent
     staleness bugs.
  2. Output folder structure now matches exactly what the notebooks
     (LoadForecastingPipeline.ipynb, ProphetGB_Pipeline.ipynb,
     LoadForecastingPipeline_R2improved.ipynb) expect:
         ./data/timeseries_2018/
         ./data/timeseries_2012/
         ./data/weather_2018/
         ./data/weather_2012/
     Previously this script saved to ./timeseries/ and ./weather/,
     which did NOT match the notebooks' DATA_DIR structure — this was
     a real path mismatch that would have caused every notebook to
     fail at Cell 6 (the data-copy step) or Cell 10/11 (file existence
     checks).
  3. Downloads BOTH 2018 and 2012 data by default, since the 2-year
     training experiments (RUN_MODE='2year') require both years.
     Set DOWNLOAD_2012 = False below if you only need the 1-year run.
  4. Metadata and selected_buildings.csv are also copied into the
     data/ folder so Cell 6's copy step in the notebooks has nothing
     left to do (it will just print "Already exists" and move on).
"""

import boto3
import pandas as pd
from botocore import UNSIGNED
from botocore.config import Config
from pathlib import Path
import shutil

# ── Setup ────────────────────────────────────────────────────
s3     = boto3.client("s3", config=Config(signature_version=UNSIGNED))
BUCKET = "oedi-data-lake"

# Two separate release paths — 2018 AMY and 2012 AMY
BASE_2018 = "nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/2022/resstock_amy2018_release_1"
BASE_2012 = "nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/2022/resstock_amy2012_release_1"

# ── Toggle: set False if you only need the 1-year experiment ──
DOWNLOAD_2012 = True

# ── Output structure — matches what the notebooks expect ──────
DATA_DIR = Path("./data")
TS_2018  = DATA_DIR / "timeseries_2018"
TS_2012  = DATA_DIR / "timeseries_2012"
WX_2018  = DATA_DIR / "weather_2018"
WX_2012  = DATA_DIR / "weather_2012"

for d in [TS_2018, TS_2012, WX_2018, WX_2012]:
    d.mkdir(parents=True, exist_ok=True)

# ── Load selected buildings ───────────────────────────────────
SELECTED_CSV = Path("./selected_buildings.csv")
if not SELECTED_CSV.exists():
    raise FileNotFoundError(
        f"{SELECTED_CSV} not found — run extractColumns.py first to "
        f"generate the building selection before downloading files."
    )

selected     = pd.read_csv(SELECTED_CSV)
building_ids = selected["bid_str"].astype(str).tolist()

# ── GISJOINs derived dynamically — NOT hardcoded ───────────────
# This is the core fix: previously this was a static list of 37 codes
# pasted into the script. Now it always matches whatever buildings
# extractColumns.py actually selected, for any random_state or zone
# quota configuration.
gisjoins = sorted(selected["in.county"].dropna().unique().tolist())

print(f"Total buildings to download : {len(building_ids)}")
print(f"Total unique GISJOINs       : {len(gisjoins)}  (derived from selected_buildings.csv)")
print(f"Download 2012 data          : {DOWNLOAD_2012}")
print("-" * 50)


def download_timeseries(year: int, base_path: str, dest_folder: Path):
    """Download building-level timeseries parquet files for a given year."""
    print(f"\n[Timeseries {year}] Downloading {len(building_ids)} files...")
    success, failed = [], []
    for i, bid in enumerate(building_ids, 1):
        key  = f"{base_path}/timeseries_individual_buildings/by_state/upgrade=0/state=CA/{bid}-0.parquet"
        dest = dest_folder / f"{bid}-0.parquet"
        if dest.exists():
            success.append(bid)
            continue
        try:
            s3.download_file(BUCKET, key, str(dest))
            success.append(bid)
        except Exception as e:
            failed.append((bid, str(e)))
        if i % 20 == 0 or i == len(building_ids):
            print(f"  [{i}/{len(building_ids)}] processed "
                  f"({len(success)} ok, {len(failed)} failed)")
    print(f"  [OK] {year}: {len(success)} downloaded, {len(failed)} failed")
    if failed:
        print(f"  Failed IDs ({year}): {[f[0] for f in failed][:10]}"
              f"{' ...' if len(failed) > 10 else ''}")
    return success, failed


def download_weather(year: int, base_path: str, dest_folder: Path):
    """Download county-level weather CSV files for a given year."""
    print(f"\n[Weather {year}] Downloading {len(gisjoins)} files...")
    success, failed = [], []
    for i, gis in enumerate(gisjoins, 1):
        key  = f"{base_path}/weather/state=CA/{gis}_{year}.csv"
        dest = dest_folder / f"{gis}_{year}.csv"
        if dest.exists():
            success.append(gis)
            continue
        try:
            s3.download_file(BUCKET, key, str(dest))
            success.append(gis)
        except Exception as e:
            failed.append((gis, str(e)))
        if i % 10 == 0 or i == len(gisjoins):
            print(f"  [{i}/{len(gisjoins)}] processed "
                  f"({len(success)} ok, {len(failed)} failed)")
    print(f"  [OK] {year}: {len(success)} downloaded, {len(failed)} failed")
    if failed:
        print(f"  Failed GISJOINs ({year}): {[f[0] for f in failed]}")
    return success, failed


# ── 1. Download 2018 (always required) ─────────────────────────
ts18_ok, ts18_fail = download_timeseries(2018, BASE_2018, TS_2018)
wx18_ok, wx18_fail = download_weather(2018, BASE_2018, WX_2018)

# ── 2. Download 2012 (required for 2-year experiments) ─────────
if DOWNLOAD_2012:
    ts12_ok, ts12_fail = download_timeseries(2012, BASE_2012, TS_2012)
    wx12_ok, wx12_fail = download_weather(2012, BASE_2012, WX_2012)
else:
    ts12_ok = ts12_fail = wx12_ok = wx12_fail = []

# ── 3. Copy metadata + selected_buildings.csv into data/ ───────
# This pre-empties Cell 6 in the notebooks (the old-folder copy step)
# so it just prints "Already exists" for everything instead of failing.
METADATA_SRC = Path("./metadata/CA_upgrade0.parquet")
METADATA_DST = DATA_DIR / "metadata" / "CA_upgrade0.parquet"
METADATA_DST.parent.mkdir(parents=True, exist_ok=True)
if METADATA_SRC.exists() and not METADATA_DST.exists():
    shutil.copy2(METADATA_SRC, METADATA_DST)
    print(f"\n[OK] Copied metadata -> {METADATA_DST}")
elif METADATA_DST.exists():
    print(f"\n[OK] Metadata already in place: {METADATA_DST}")
else:
    print(f"\n[WARNING] Metadata source not found at {METADATA_SRC} -- "
          f"copy CA_upgrade0.parquet into {METADATA_DST} manually before running notebooks.")

# selected_buildings.csv is already at repo root, which is exactly
# where the notebooks' SELECTED = BASE_DIR / 'selected_buildings.csv' expects it.

# ── Summary ───────────────────────────────────────────────────
print("\n" + "=" * 50)
print("DOWNLOAD SUMMARY")
print("=" * 50)
print(f"2018 timeseries : {len(ts18_ok)}/{len(building_ids)} successful")
print(f"2018 weather    : {len(wx18_ok)}/{len(gisjoins)} successful")
if DOWNLOAD_2012:
    print(f"2012 timeseries : {len(ts12_ok)}/{len(building_ids)} successful")
    print(f"2012 weather    : {len(wx12_ok)}/{len(gisjoins)} successful")
print(f"\nData saved to   : {DATA_DIR.resolve()}")
print(f"  {TS_2018}")
print(f"  {TS_2012}" if DOWNLOAD_2012 else "  (2012 timeseries skipped)")
print(f"  {WX_2018}")
print(f"  {WX_2012}" if DOWNLOAD_2012 else "  (2012 weather skipped)")
print(f"\nNext step: open LoadForecastingPipeline.ipynb (or the R2-improved")
print(f"version) -- Cell 6's old-folder copy step will now be a no-op since")
print(f"all files are already in the expected ./data/ location.")
