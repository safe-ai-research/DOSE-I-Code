from pathlib import Path
import numpy as np
import re
import json

def summarize(experimentname, model):
    experimentname = experimentname
    model = model
    folderpath = Path(fr"{model}\{experimentname}")
    output_file = folderpath / f"{experimentname}_metric_results.txt"

    metrics = {
        "Cross-Entropy Loss": [],
        "AUC-ROC": [],
        "Accuracy": [],
        "Precision": [],
        "Recall": [],
        "F1 Score": [],
        "Percentage monotone Class labels": []
    }
    # To store file counts per subfolder
    file_counts = []

    # Read metrics from each subfolder
    if model == "forest":
        num = 2
    else:
        num = 1
    for subfolder in folderpath.iterdir():
        if subfolder.is_dir():
            # Count all files in the subfolder
            num_files = sum(1 for item in subfolder.iterdir() if item.is_file()) - num
            file_counts.append(num_files)

            results_file = subfolder / f"ModelFinalResults-{model}.txt"
            if results_file.exists():
                try:
                    with open(results_file, 'r') as file:
                        content = file.read()
                        for line in content.split('\n'):
                            if ":" in line:
                                parts = line.split(':')
                                if len(parts) < 3:
                                    continue  # malformed line
                                key = parts[1].strip().split('Median ')[-1]
                                raw_value = parts[2].strip()

                                try:
                                    if raw_value.lower() == 'nan':
                                        value = float('nan')
                                    else:
                                        match = re.search(r"[-+]?\d*\.\d+|\d+", raw_value)
                                        if match:
                                            value = float(match.group())
                                        else:
                                            continue  # no valid number found

                                    if key in metrics:
                                        metrics[key].append(value)

                                except Exception as val_err:
                                    print(f"Error parsing value in line: {line} — {val_err}")

                except Exception as e:
                    print(f"Error processing {results_file}: {e}")

    # Compute mean, std, and 95% CI
    results = {}

    for k, values in metrics.items():
        clean_values = [v for v in values if not np.isnan(v)]
        num_nans = len(values) - len(clean_values)

        if clean_values:
            mean = np.mean(clean_values)
            std = np.std(clean_values, ddof=1)
            n = len(clean_values)
            margin = 1.96 * std / np.sqrt(n)
            ci_lower = mean - margin
            ci_upper = mean + margin
            results[k] = {
                "mean": mean,
                "std": std,
                "ci": [ci_lower, ci_upper],
                "n": n,
                "num_nans": num_nans
            }
        else:
            results[k] = {
                "mean": None,
                "std": None,
                "ci": None,
                "n": 0,
                "num_nans": len(values)
            }

    # Calculate average file count
    average_file_count = np.mean(file_counts) if file_counts else 0
    output_file_json = folderpath / f"{experimentname}_metric_results.json"
    with open(output_file_json, "w") as fp:
        json.dump(metrics, fp)
        # Write results to file
    with open(output_file, 'w') as f:
        for metric, stats in results.items():
            f.write(f"{metric}:\n")
            if stats["mean"] is not None:
                f.write(f"  Mean: {stats['mean']:.4f}\n")
                f.write(f"  Std: {stats['std']:.4f}\n")
                f.write(f"  95% CI: [{stats['ci'][0]:.4f}, {stats['ci'][1]:.4f}] (n={stats['n']})\n")
            else:
                f.write("Only NaN\n")

            if stats['num_nans'] > 0:
                f.write(f"  NaN entries: {stats['num_nans']}\n")
            f.write("\n")

        f.write(f"Average number of splits per run: {average_file_count:.2f}\n")

    print(f"Results with confidence intervals saved to: {output_file}")
