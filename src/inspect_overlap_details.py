
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

inpatient["PREVIOUS_STOP"] = (
    inpatient.groupby("PATIENT")["STOP"].shift(1)
)
inpatient["PREVIOUS_ID"] = (
    inpatient.groupby("PATIENT")["Id"].shift(1)
)

overlaps = inpatient[
    inpatient["PREVIOUS_STOP"].notna()
    & (inpatient["START"] < inpatient["PREVIOUS_STOP"])
].copy()

details = [
    "PATIENT", "Id", "PREVIOUS_ID",
    "START", "STOP", "ENCOUNTERCLASS",
    "CODE", "DESCRIPTION", "ORGANIZATION",
    "PROVIDER"
]

print(overlaps[details].to_string(index=False))
