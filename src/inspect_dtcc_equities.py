import argparse
import csv
from pathlib import Path


def text_value(row, column):
    return (row.get(column) or "").strip()


def contains_jpm(row):
    fields = [
        "Underlying Asset Name",
        "UPI Underlier Name",
        "Underlier ID-Leg 1",
        "Underlier ID-Leg 2",
        "UPI FISN",
    ]
    joined_text = " | ".join(text_value(row, field) for field in fields).upper()
    return "JPM" in joined_text or "JPMORGAN" in joined_text or "J.P. MORGAN" in joined_text


parser = argparse.ArgumentParser()
parser.add_argument("--file", required=True, help="DTCC CSV file path")
args = parser.parse_args()

csv_path = Path(args.file)

total_rows = 0
jpm_rows = 0
missing_premium = 0
missing_strike = 0
non_standardized_rows = 0
samples = []

with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
    reader = csv.DictReader(file)

    for row in reader:
        total_rows += 1

        if not contains_jpm(row):
            continue

        jpm_rows += 1

        if text_value(row, "Option Premium Amount") == "":
            missing_premium += 1

        if text_value(row, "Strike Price") == "":
            missing_strike += 1

        if text_value(row, "Non-standardized term indicator").upper() == "TRUE":
            non_standardized_rows += 1

        if len(samples) < 10:
            samples.append(
                {
                    "execution_time": text_value(row, "Execution Timestamp"),
                    "underlier": text_value(row, "UPI Underlier Name"),
                    "premium": text_value(row, "Option Premium Amount"),
                    "strike": text_value(row, "Strike Price"),
                    "option_type": text_value(row, "Option Type"),
                    "option_style": text_value(row, "Option Style"),
                    "first_exercise_date": text_value(row, "First exercise date"),
                    "non_standardized": text_value(
                        row, "Non-standardized term indicator"
                    ),
                }
            )

print(f"Total rows scanned: {total_rows:,}")
print(f"JPM-related rows: {jpm_rows:,}")
print(f"JPM rows missing premium: {missing_premium:,}")
print(f"JPM rows missing strike: {missing_strike:,}")
print(f"JPM non-standardized rows: {non_standardized_rows:,}")

print("\nFirst JPM-related records:")
for index, sample in enumerate(samples, start=1):
    print(f"\nRecord {index}")
    for key, value in sample.items():
        print(f"  {key}: {value}")