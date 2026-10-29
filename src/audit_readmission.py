
import pandas as pd

patients = pd.read_csv("data/raw/patients.csv")
encounters = pd.read_csv("data/raw/encounters.csv")

encounters["START"] = pd.to_datetime(
    encounters["START"], utc=True, errors="coerce"
)
encounters["STOP"] = pd.to_datetime(
    encounters["STOP"], utc=True, errors="coerce"
)

inpatient = encounters[
    encounters["ENCOUNTERCLASS"].str.lower() == "inpatient"
].copy()

inpatient = inpatient.sort_values(["PATIENT", "START"])

print("Total inpatient encounters:", len(inpatient))
print("Patients with inpatient encounters:", inpatient["PATIENT"].nunique())

# Check duration and chronology
duration = inpatient["STOP"] - inpatient["START"]

print("\n--- INPATIENT DATE CHECKS ---")
print("Stop before start:", (duration < pd.Timedelta(0)).sum())
print("Zero-duration encounters:", (duration == pd.Timedelta(0)).sum())
print("Median encounter duration:", duration.median())

# Examine time between consecutive inpatient encounters
inpatient["PREVIOUS_STOP"] = (
    inpatient.groupby("PATIENT")["STOP"].shift(1)
)

inpatient["GAP_DAYS"] = (
    inpatient["START"] - inpatient["PREVIOUS_STOP"]
).dt.total_seconds() / 86400

# A subsequent inpatient encounter within 30 days of the
# previous encounter's recorded stop time.
followup = inpatient["PREVIOUS_STOP"].notna()
gaps = inpatient.loc[followup, "GAP_DAYS"]

print("\n--- REPEAT INPATIENT ENCOUNTERS ---")
print("Patients with multiple inpatient encounters:",
      inpatient.groupby("PATIENT").size().gt(1).sum())
print("Subsequent inpatient encounters:", len(gaps))
print("Same-day or overlapping encounters:", (gaps <= 0).sum())
print("Within 30 days:", ((gaps > 0) & (gaps <= 30)).sum())
print("31-90 days:", ((gaps > 30) & (gaps <= 90)).sum())
print("Over 90 days:", (gaps > 90).sum())
print("\nGap statistics:")
print(gaps.describe())

# Demographic availability
print("\n--- DEMOGRAPHICS ---")
print("Missing birth dates:", patients["BIRTHDATE"].isna().sum())
print("Missing gender:", patients["GENDER"].isna().sum())
print("Missing race:", patients["RACE"].isna().sum())
print("Missing ethnicity:", patients["ETHNICITY"].isna().sum())
