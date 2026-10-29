
import pandas as pd

patients = pd.read_csv("data/raw/patients.csv")
encounters = pd.read_csv("data/raw/encounters.csv")

# 1. Encounter type distribution
print("\n--- ENCOUNTER CLASSES ---")
print(encounters["ENCOUNTERCLASS"].value_counts(dropna=False))

# 2. Encounter date range
encounters["START"] = pd.to_datetime(
    encounters["START"], errors="coerce"
)
encounters["STOP"] = pd.to_datetime(
    encounters["STOP"], errors="coerce"
)

print("\n--- DATE RANGE ---")
print("Earliest encounter:", encounters["START"].min())
print("Latest encounter:", encounters["START"].max())
print("Missing start dates:", encounters["START"].isna().sum())
print("Missing stop dates:", encounters["STOP"].isna().sum())

# 3. Patient linkage
print("\n--- PATIENT LINKAGE ---")
print("Unique patient IDs:", patients["Id"].nunique())
print("Unique encounter patients:", encounters["PATIENT"].nunique())

unknown_patients = ~encounters["PATIENT"].isin(patients["Id"])
print("Encounters with unknown patient IDs:", unknown_patients.sum())

# 4. Duplicate records
print("\n--- DUPLICATES ---")
print("Duplicate patient IDs:", patients["Id"].duplicated().sum())
print("Duplicate encounter IDs:", encounters["Id"].duplicated().sum())

# 5. Patient encounter counts
print("\n--- ENCOUNTERS PER PATIENT ---")
counts = encounters.groupby("PATIENT").size()
print(counts.describe())
print("Patients with multiple encounters:", (counts > 1).sum())
