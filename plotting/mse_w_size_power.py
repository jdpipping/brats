import os, sys
import pandas as pd
import matplotlib.pyplot as plt
from BRAT.utils import plot_mean_std_trajectories
import matplotlib.ticker as mticker
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import MaxNLocator, LogLocator

plt.style.use('../matplotlibrc')

base_dir = '../reports/mse_results'
size_power_csv = '../submission/size_and_power.csv'

datasets = {
    186: "Wine Quality",
    544: "Obesity",
    360: "Air Quality"
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

fig, axes = plt.subplots(
    nrows=1, ncols=4,
    figsize=(16, 3),
    sharey=False,
    dpi=110
)

df_sp = pd.read_csv(size_power_csv)
ax0   = axes[0]

ax0.plot(df_sp['n'], df_sp['type_I_error'], label='Type I Error')
ax0.plot(df_sp['n'], df_sp['type_II_error'], label='Type II Error')

ax0.set_xlabel("Training Samples", fontsize=8, labelpad=3)
ax0.set_ylabel("Error", fontsize=8, labelpad=3)
ax0.set_yscale("linear")
ax0.tick_params(labelsize=5)
ax0.set_title("Type I & Type II Error", fontsize=8, pad=3)
ax0.legend(fontsize=6)
ax0.get_xaxis().get_offset_text().set_visible(False)
ax0.get_yaxis().get_offset_text().set_visible(False)

for ax, (ds_id, title) in zip(axes[1:], datasets.items()):
    csv_file = os.path.join(base_dir, f"{ds_id}.csv")
    if not os.path.exists(csv_file):
        raise FileNotFoundError(csv_file)

    mse_runs, n_epoch = csv_to_mse_runs(csv_file)

    plot_mean_std_trajectories(
        mse_runs,
        n_epoch,
        dataset_id=ds_id,
        title=title,
        ax=ax,
        fontsize=7
    )

    ax.set_yscale("log")
    yfmt = ScalarFormatter()
    yfmt.set_scientific(False)
    yfmt.set_useOffset(False)
    ax.yaxis.set_major_formatter(yfmt)
    xfmt = mticker.FormatStrFormatter('%.2f')
    ax.xaxis.set_major_formatter(xfmt)
    # integer locator
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    # integer formatter
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=3))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))
    ax.tick_params(labelsize=5)
    ax.set_title(title, fontsize=8, pad=3)

fig.canvas.draw()
for ax in axes:
    ax.get_xaxis().get_offset_text().set_visible(False)
    ax.get_yaxis().get_offset_text().set_visible(False)

fig.tight_layout(w_pad=1.2)
plt.show()
