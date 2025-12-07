# make_fitbit_aggregates_and_augment.py
import pandas as pd
import numpy as np
from pathlib import Path
import zipfile


fitbit_path = Path("data/raw/daily_fitbit_sema_df_unprocessed.csv")
obesity_path = Path("data/raw/ObesityDataSet_raw_and_data_sinthetic.csv")  

out_dir = Path("data/processed")
out_dir.mkdir(parents=True, exist_ok=True)

per_user_path = out_dir / "per_user_fitbit_aggregates.csv"
aug_path = out_dir / "obesity_fitbit_augmented.csv"
zip_path = out_dir / "fitbit_obesity_files.zip"

# ---- load ----
print("Loading files...")
fitbit = pd.read_csv(fitbit_path)
obesity = pd.read_csv(obesity_path)

# ---- clean names ----
fitbit.columns = [c.strip() for c in fitbit.columns]
obesity.columns = [c.strip() for c in obesity.columns]

# ---- candidate columns to aggregate ----
candidate_cols = [
    "steps", "caloriesOut", "veryActiveMinutes", "fairlyActiveMinutes",
    "lightlyActiveMinutes", "sedentaryMinutes", "totalMinutesAsleep",
    "totalTimeInBed", "sleep_points_percentage", "stress_score", "rmssd"
]
cols_present = [c for c in candidate_cols if c in fitbit.columns]
print("Found Fitbit columns:", cols_present)

# ---- date conversion if present ----
if "date" in fitbit.columns:
    fitbit["date"] = pd.to_datetime(fitbit["date"], errors="coerce")

# ---- ensure id exists (if not, create single group) ----
if "id" not in fitbit.columns:
    print("No 'id' column in Fitbit; creating single-group aggregate (id=0)")
    fitbit["id"] = 0

# ---- per-user aggregation ----
agg_funcs = {c: ["mean","median","std","min","max","count"] for c in cols_present}
per_user_agg = fitbit.groupby("id").agg(agg_funcs)
per_user_agg.columns = ["_".join(col).strip() for col in per_user_agg.columns.values]
per_user_agg = per_user_agg.reset_index()

# ---- derived metric: sleep efficiency (mean per user) ----
if "totalMinutesAsleep" in fitbit.columns and "totalTimeInBed" in fitbit.columns:
    fitbit["sleep_efficiency"] = fitbit["totalMinutesAsleep"] / fitbit["totalTimeInBed"].replace({0: np.nan})
    sleep_eff_mean = fitbit.groupby("id")["sleep_efficiency"].mean().reset_index().rename({"sleep_efficiency":"sleep_efficiency_mean"}, axis=1)
    per_user_agg = per_user_agg.merge(sleep_eff_mean, on="id", how="left")

# ---- save per-user aggregates ----
per_user_agg.to_csv(per_user_path, index=False)
print("Saved per-user aggregates to:", per_user_path)

# ---- population-level aggregates (from per-user means when present) ----
pop_aggregates = {}
for c in cols_present:
    mean_col = f"{c}_mean"
    if mean_col in per_user_agg.columns:
        pop_aggregates[f"fit_{c}_mean"] = float(per_user_agg[mean_col].mean())
    else:
        pop_aggregates[f"fit_{c}_mean"] = float(fitbit[c].mean())
# optional: include sleep_efficiency_mean
if "sleep_efficiency_mean" in per_user_agg.columns:
    pop_aggregates["fit_sleep_efficiency_mean"] = float(per_user_agg["sleep_efficiency_mean"].mean())

print("Population-level aggregates computed:", {k: round(v,2) for k,v in list(pop_aggregates.items())[:8]})

# ---- add population aggregates as constant cols to obesity ----
obesity_aug = obesity.copy()
for k,v in pop_aggregates.items():
    obesity_aug[k] = v

# ---- save augmented obesity dataset ----
obesity_aug.to_csv(aug_path, index=False)
print("Saved obesity augmented file to:", aug_path)

# ---- make zip ----
with zipfile.ZipFile(zip_path, "w") as z:
    z.write(per_user_path, arcname=per_user_path.name)
    z.write(aug_path, arcname=aug_path.name)

print("Created ZIP:", zip_path)
print("Done.")
