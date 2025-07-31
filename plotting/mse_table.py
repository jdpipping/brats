from pathlib import Path
import pandas as pd

ROOT = Path("../reports/mse_results")

files = sorted(ROOT.rglob("*.csv"))

all_rows = []

for csv_path in files:
    try:
        df = pd.read_csv(csv_path)
        df["Epoch"] = pd.to_numeric(df["Epoch"], errors="coerce")
        df["Run_ID"] = pd.to_numeric(df["Run_ID"], errors="coerce")
        df["MSE"]   = pd.to_numeric(df["MSE"],   errors="coerce")
        df = df.dropna(subset=["Model", "Run_ID", "Epoch", "MSE"])

        idx = df.groupby(["Model", "Run_ID"])["Epoch"].idxmax()
        finals = df.loc[idx]

        summary = (
            finals.groupby("Model")["MSE"]
            .agg(mean_final_mse="mean", std_final_mse="std", n_runs="count")
            .sort_values("mean_final_mse")
            .reset_index()
        )

        # Print to console
        print(f"\n=== {csv_path.name} ===")
        print(summary.to_string(index=False))

        # Save next to the CSV
        out_path = csv_path.with_name(csv_path.stem + "__final_mean_mse.csv")
        summary.to_csv(out_path, index=False)

        # Keep for a combined file
        summary = summary.assign(source_file=csv_path.name)
        all_rows.append(summary[["source_file", "Model", "mean_final_mse", "std_final_mse", "n_runs"]])

    except Exception as e:
        print(f"[skip] {csv_path.name}: {e}")

if all_rows:
    combined = pd.concat(all_rows, ignore_index=True)
    combined.to_csv(ROOT / "final_mean_mse__ALL.csv", index=False)
    wide = combined.pivot_table(index="source_file", columns="Model", values="mean_final_mse", aggfunc="first")
    wide.to_csv(ROOT / "final_mean_mse__ALL__wide.csv")
    print("\nSaved combined summaries: final_mean_mse__ALL.csv and final_mean_mse__ALL__wide.csv")
else:
    print("\nNo valid CSVs summarized.")