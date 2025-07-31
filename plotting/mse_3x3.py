import os, sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.ticker import ScalarFormatter, MaxNLocator, LogLocator
from BRAT.utils import plot_mean_std_trajectories

plt.style.use('../matplotlibrc')

base_dir = '../reports/mse_results'

datasets = {
    186: "Wine Quality",
    544: "Obesity",
    360: "Air Quality",
    1: "Abalone",
    925: "Infared Thermography Temperature",
    183: "Communities and Crime",
    890: "AIDS Clinical Trials Group Study 175",
    10: "Automobile",
    320: "Student Performance"
}


def csv_to_mse_runs(csv_path):
    df = pd.read_csv(csv_path)
    n_runs  = df["Run_ID"].nunique()
    models  = df["Model"].unique()
    n_epoch = df["Epoch"].max()

    mse_runs = []
    for run_id in range(1, n_runs + 1):
        run_df = df[df["Run_ID"] == run_id]
        run_dict = {
            model: run_df[run_df["Model"] == model]
                        .sort_values("Epoch")["MSE"]
                        .to_numpy()
            for model in models
        }
        mse_runs.append(run_dict)
    return mse_runs, n_epoch

def apply_current_formatting(ax, title):
    ax.set_yscale("log")

    yfmt = ScalarFormatter()
    yfmt.set_scientific(False)
    yfmt.set_useOffset(False)
    ax.yaxis.set_major_formatter(yfmt)

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))

    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=3))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))

    ax.tick_params(labelsize=5)
    ax.set_title(title, fontsize=8, pad=3)

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10, 10), sharey=False, dpi=110)
axes_flat = axes.ravel()

for ax, (ds_id, title) in zip(axes_flat, datasets.items()):
    csv_file = os.path.join(base_dir, f"{ds_id}.csv")
    if not os.path.exists(csv_file):
        ax.text(0.5, 0.5, f"Missing CSV:\n{ds_id}.csv", ha='center', va='center', fontsize=7)
        ax.set_axis_off()
        continue

    mse_runs, n_epoch = csv_to_mse_runs(csv_file)

    plot_mean_std_trajectories(
        mse_runs,
        n_epoch,
        dataset_id=ds_id,
        title=title,
        ax=ax,
        fontsize=7
    )

    apply_current_formatting(ax, title)

for ax in axes_flat[len(datasets):]:
    ax.set_axis_off()

fig.canvas.draw()
for ax in axes_flat[:len(datasets)]:
    if ax.has_data():
        ax.get_xaxis().get_offset_text().set_visible(False)
        ax.get_yaxis().get_offset_text().set_visible(False)

fig.tight_layout(w_pad=1.2, h_pad=1.2)
plt.show()
