import pandas as pd

# ── Step 1: Load metadata ─────────────────────────────────────
meta = pd.read_parquet("./metadata/CA_upgrade0.parquet")

# ── Step 2: Load available S3 file IDs ───────────────────────
# Parse available_files.txt to get all building IDs actually on S3
with open("./available_files.txt") as f:
    lines = f.readlines()

available_ids = set()
for line in lines:
    parts = line.strip().split()
    if not parts:
        continue
    filename = parts[-1]
    if filename.endswith("-0.parquet"):
        bid = filename.replace("-0.parquet", "")
        available_ids.add(bid)

print(f"Total building files available on S3: {len(available_ids)}")

# ── Step 3: Add a column to metadata showing S3 availability ─
# The metadata uses bldg_id — convert to string for matching
meta["bid_str"] = meta["bldg_id"].astype(str)
meta["on_s3"]   = meta["bid_str"].isin(available_ids)

print(f"Metadata buildings found on S3: {meta['on_s3'].sum()}")

# ── Step 4: Filter to valid, downloadable buildings ──────────
meta_filtered = meta[
    (meta["on_s3"] == True) &
    (meta["in.geometry_building_type_recs"] == "Single-Family Detached") &
    (meta["in.has_pv"] == "No")
]

print(f"\nAfter filtering (on S3 + Single Family + No PV): {len(meta_filtered)} buildings")
print("\nAvailable per climate zone:")
print(meta_filtered["in.ashrae_iecc_climate_zone_2004"].value_counts())

# ── Step 5: Stratified selection ─────────────────────────────
selection_plan = {
    "3B": 35,   # plenty available
    "3C": 25,   # plenty available
    "4B": 15,   # moderate
    "4C": 10,   # limited
    "5B": 10,   # limited
    "2B":  5,   # most limited — take what we can
}
# Total: 100 buildings

selected = []

for zone, n in selection_plan.items():
    subset = meta_filtered[
        meta_filtered["in.ashrae_iecc_climate_zone_2004"] == zone
    ]

    if len(subset) == 0:
        print(f"WARNING: Zone {zone} has 0 valid buildings — skipping")
        continue

    if len(subset) < n:
        print(f"WARNING: Zone {zone} only has {len(subset)} — taking all")
        n = len(subset)

    sampled = subset.sample(n=n, random_state=42).copy()
    sampled["climate_zone"] = zone
    selected.append(sampled)
    print(f"Zone {zone}: selected {n} from {len(subset)} available")

final = pd.concat(selected).reset_index(drop=True)

# ── Step 6: Save ─────────────────────────────────────────────
cols_to_keep = [
    "bldg_id",
    "bid_str",             # S3-matched ID string
    "in.county",           # GISJOIN → links to weather file
    "in.county_name",
    "in.ashrae_iecc_climate_zone_2004",
    "in.geometry_building_type_recs",
    "in.geometry_floor_area",
    "in.bedrooms",
    "in.vintage",
    "in.hvac_cooling_type",
    "climate_zone"
]

final[cols_to_keep].to_csv("./selected_buildings.csv", index=False)

print(f"\n{'='*50}")
print(f"Total selected: {len(final)} buildings")
print(final.groupby("climate_zone")["bldg_id"].count())
print(f"\nUnique GISJOIN codes needed for weather download:")
print(final["in.county"].unique())
print(f"\n✓ saved to ./selected_buildings.csv")