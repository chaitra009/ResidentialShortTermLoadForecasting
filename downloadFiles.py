import boto3
import pandas as pd
from botocore import UNSIGNED
from botocore.config import Config
from pathlib import Path

# ── Setup ────────────────────────────────────────────────────
s3     = boto3.client("s3", config=Config(signature_version=UNSIGNED))
BUCKET = "oedi-data-lake"
BASE   = "nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/2022/resstock_amy2018_release_1"

Path("./timeseries").mkdir(parents=True, exist_ok=True)
Path("./weather").mkdir(parents=True, exist_ok=True)

# ── Load selected buildings ───────────────────────────────────
selected     = pd.read_csv("./selected_buildings.csv")
building_ids = selected["bid_str"].astype(str).tolist()
gisjoins     = [
    'G0600990', 'G0600370', 'G0600130', 'G0600710', 'G0600590', 'G0600650',
    'G0600730', 'G0601130', 'G0600290', 'G0601010', 'G0600190', 'G0600310',
    'G0600610', 'G0600850', 'G0601110', 'G0600450', 'G0600790', 'G0600750',
    'G0600530', 'G0600010', 'G0600970', 'G0600810', 'G0600550', 'G0600870',
    'G0600330', 'G0600050', 'G0600270', 'G0600170', 'G0600430', 'G0600230',
    'G0600150', 'G0600630', 'G0600350', 'G0600570', 'G0600930', 'G0600490',
    'G0600250'
]

print(f"Total buildings to download : {len(building_ids)}")
print(f"Total weather files         : {len(gisjoins)}")
print("-" * 50)

# ── 1. Download timeseries files ──────────────────────────────
print("\n[1/2] Downloading timeseries files...")
success, failed = [], []

for i, bid in enumerate(building_ids, 1):
    key  = f"{BASE}/timeseries_individual_buildings/by_state/upgrade=0/state=CA/{bid}-0.parquet"
    dest = f"./timeseries/{bid}-0.parquet"

    if Path(dest).exists():
        print(f"  [{i}/{len(building_ids)}] ✓ Already exists: {bid}")
        success.append(bid)
        continue

    try:
        s3.download_file(BUCKET, key, dest)
        print(f"  [{i}/{len(building_ids)}] ✓ Downloaded: {bid}")
        success.append(bid)
    except Exception as e:
        print(f"  [{i}/{len(building_ids)}] ✗ Failed: {bid} → {e}")
        failed.append(bid)

print(f"\nTimeseries: {len(success)} downloaded, {len(failed)} failed")
if failed:
    print(f"Failed IDs: {failed}")

# ── 2. Download weather files ─────────────────────────────────
print("\n[2/2] Downloading weather files...")
wx_success, wx_failed = [], []

for i, gis in enumerate(gisjoins, 1):
    # Actual filename format: G0600010_2018.csv (not G0600010.csv)
    key  = f"{BASE}/weather/state=CA/{gis}_2018.csv"
    dest = f"./weather/{gis}_2018.csv"

    if Path(dest).exists():
        print(f"  [{i}/{len(gisjoins)}] ✓ Already exists: {gis}")
        wx_success.append(gis)
        continue

    try:
        s3.download_file(BUCKET, key, dest)
        print(f"  [{i}/{len(gisjoins)}] ✓ Downloaded: {gis}")
        wx_success.append(gis)
    except Exception as e:
        print(f"  [{i}/{len(gisjoins)}] ✗ Failed: {gis} → {e}")
        wx_failed.append(gis)

print(f"\nWeather: {len(wx_success)} downloaded, {len(wx_failed)} failed")
if wx_failed:
    print(f"Failed GISJOINs: {wx_failed}")

# ── Summary ───────────────────────────────────────────────────
print("\n" + "=" * 50)
print("DOWNLOAD SUMMARY")
print("=" * 50)
print(f"Timeseries : {len(success)}/{len(building_ids)} successful")
print(f"Weather    : {len(wx_success)}/{len(gisjoins)} successful")
print(f"Total files: {len(success) + len(wx_success)}")
print(f"\nNext step: run preprocessPipeline.py")