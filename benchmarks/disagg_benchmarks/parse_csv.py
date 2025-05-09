import os
import json
import csv

# List of fields to extract
fields = [
    "completed",
    "total_input_tokens",
    "total_output_tokens",
    "request_throughput",
    "output_throughput",
    "total_token_throughput",
    "mean_ttft_ms",
    "median_ttft_ms",
    "std_ttft_ms",
    "p99_ttft_ms",
    "mean_tpot_ms",
    "median_tpot_ms",
    "std_tpot_ms",
    "p99_tpot_ms",
    "mean_itl_ms",
    "median_itl_ms",
    "std_itl_ms",
    "p99_itl_ms"
]

def extract_fields(obj, fields):
    """Extracts the specified fields from a dict (top-level only)."""
    return [obj.get(field, "") for field in fields]

root_dir = "results_1p2d_sharegpt"
csv_output = "results_1p2d_sharegpt/parsed_results_1p2d_sharegpt_statistics.csv"

with open(csv_output, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["file_path"] + fields)  # CSV header

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".json"):
                file_path = os.path.join(dirpath, filename)
                try:
                    with open(file_path, "r") as f:
                        data = json.load(f)
                    # Only extract fields at the top level
                    row = [file_path] + extract_fields(data, fields)
                    writer.writerow(row)
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
