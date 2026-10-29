
import pandas as pd

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

inpatient["PREVIOUS_ID"] = (
    inpatient.groupby("PATIENT")["Id"].shift(1)
)
inpatient["PREVIOUS_START"] = (
    inpatient.groupby("PATIENT")["START"].shift(1)
)
inpatient["PREVIOUS_STOP"] = (
    inpatient.groupby("PATIENT")["STOP"].shift(1)
)

inpatient["GAP_HOURS"] = (
    inpatient["START"] - inpatient["PREVIOUS_STOP"]
).dt.total_seconds() / 3600

overlaps = inpatient[
    inpatient["PREVIOUS_STOP"].notna()
    & (inpatient["GAP_HOURS"] <= 0)
]

columns = [
    "PATIENT", "PREVIOUS_ID", "Id",
    "PREVIOUS_START", "PREVIOUS_STOP",
    "START", "STOP", "GAP_HOURS"
]

print(overlaps[columns].to_string(index=False))
print("\nNumber of overlapping or same-time records:", len(overlaps))
