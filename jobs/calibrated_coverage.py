import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
# Prevent ptitprince from raising cmap errors
cm.register_cmap = lambda *args, **kwargs: None

from tqdm import tqdm
from itertools import product
from BRAT.brat import BRATD
from BRAT.utils import generate_data, find_min_scale
import ptitprince as pt

from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
plt.style.use('matplotlibrc')

mp.set_start_method('fork', force=True)
def calibrate_scale(widths_cal, y_cal, y_pred_cal, alpha=0.05):
    """
    Two-stage search for global c so that empirical coverage ≈ 1-alpha.
    """
    n = len(y_cal)
    def cov(c):
        lo = y_pred_cal - c * widths_cal
        hi = y_pred_cal + c * widths_cal
        return np.mean((y_cal >= lo) & (y_cal <= hi))

    # exponential search
    c = 1.0
    if cov(c) < 1 - alpha:
        while cov(c) < 1 - alpha:
            c *= 2
    C = c

    # binary refine on [0, C]
    lo, hi = 0.0, C
    tol = 2.0 / n
    while hi - lo > tol:
        mid = (lo + hi) / 2
        if cov(mid) < 1 - alpha:
            lo = mid
        else:
            hi = mid
    return hi

# Hyperparameter grid

lr_values = [0.3, 0.6, 1.0]
dr_values = [0.3, 0.6, 0.9]
sr_values = [0.8]
max_depth_values = [4,6,8]

# Fixed settings
num_test_pts = 200
rep = 50
epoch = 200
Nystrom_subsample = 0.1
in_bag = False
function = 'friedman1'

# Create directories for saving results
os.makedirs("../reports", exist_ok=True)
os.makedirs("../plots/coverage_rate", exist_ok=True)

# Iterate over all combinations of hyperparameters
for idx, (lr, dr, sr, max_depth) in enumerate(product(lr_values, dr_values, sr_values, max_depth_values)):
    # Data generation
    X_train, y_train, X_test, y_test, y_test_true, X_cal, y_cal = generate_data(
        function_type=function, 
        n_train=1000, 
        n_test=200,
        n_calibration=200, 
        noise_std=0.1,
        seed=idx
    )
    test_points = X_test[:num_test_pts]
    y_true = y_test_true[:num_test_pts]
    y_obs = y_test[:num_test_pts]

    def training_rep(k):
        # Training repetitions
        bratd = BRATD(
            n_estimators=epoch,
            learning_rate=lr,
            subsample_rate=sr,
            dropout_rate=dr,
            max_depth=max_depth,
            min_samples_split=2,
            disable_tqdm=True
        )
        
        # Data generation
        X_train, y_train, X_test, y_test, y_test_true, X_cal, y_cal = generate_data(
            function_type=function, 
            n_train=1000, 
            n_test=200,
            n_calibration=200, 
            noise_std=0.1, 
            seed=k
        ) # setting seed to be k. this means that every hyperparameter combination has the same set of seeds.
        test_points = X_test[:num_test_pts]
        y_true = y_test_true[:num_test_pts]
        y_obs = y_test[:num_test_pts]
        
        bratd.fit(X_train, y_train, X_test, y_test)
        _, _, _ = bratd.unif_nystrom(Nystrom_subsample)
        bratd.sketch_K()
        sigma_hat2 = bratd.est_sigma_hat2(in_bag)
        sigma_hat = np.sqrt(sigma_hat2)
        lam = bratd.learning_rate
        q = 1 - bratd.dropout_rate
        s = (1 + lam * q) / lam

        # 1) Build calibration arrays by iterating
        y_pred_cal = []
        tau_cal    = []
        pi_sigma_cal = []
        for x_cal in X_cal:
            y_pc = bratd.predict(x_cal.reshape(1, -1)).item()
            rn_c = bratd.sketch_r(x_cal.reshape(1, -1)).item()
            t_c  = s * rn_c * sigma_hat
            p_c  = np.sqrt(sigma_hat2 + t_c**2)

            y_pred_cal.append(y_pc)
            tau_cal.append(t_c)
            pi_sigma_cal.append(p_c)

        y_pred_cal    = np.array(y_pred_cal)    # shape (n_cal,)
        widths_pi_cal = np.array(pi_sigma_cal)  # shape (n_cal,)

        # 2) Calibrate once per interval type
        c = calibrate_scale(widths_pi_cal, y_cal, y_pred_cal)

        for i, x in enumerate(test_points):
            y_pred = bratd.predict(x.reshape(1, -1))
            rn_norm = bratd.sketch_r(x.reshape(1, -1))
            tau_hat = s * rn_norm * sigma_hat
            pi_sigma = np.sqrt(sigma_hat2 + tau_hat**2)

            y_pred_cal = bratd.predict(X_cal)

            ci_lower = y_pred - tau_hat
            ci_upper = y_pred + tau_hat
            ci_width = ci_upper - ci_lower
            ci_covered = int(ci_lower.item() <= y_true[i] <= ci_upper.item())

            pi_lower = y_pred - pi_sigma
            pi_upper = y_pred + pi_sigma
            pi_width = pi_upper - pi_lower
            ri_lower = y_pred - np.sqrt(2) * tau_hat
            ri_upper = y_pred + np.sqrt(2) * tau_hat
            ri_width = ri_upper - ri_lower

            y_pred_cal = bratd.predict(X_cal)
            res_cal = np.abs(y_pred_cal - y_cal)
            qtle_hat = np.quantile(res_cal, 1 - 0.05)
            cfml_lower = y_pred - qtle_hat
            cfml_upper = y_pred + qtle_hat
            cfml_covered = int(cfml_lower.item() <= y_obs[i] <= cfml_upper.item())
            pi_covered = int(pi_lower.item() <= y_obs[i] <= pi_upper.item())

            ci_lower_scaled, ci_upper_scaled = y_pred- c * ci_width/2, y_pred + c * ci_width/2
            pi_lower_scaled, pi_upper_scaled = y_pred- c * pi_width/2, y_pred + c * pi_width/2
            ri_lower_scaled, ri_upper_scaled = y_pred- c * ri_width/2, y_pred + c * ri_width/2
            
            return {
                "model_idx": k,
                "pt_idx": i,
                "y_true": y_true[i].item(),
                "y_obs": y_obs[i].item(),
                "y_pred": y_pred.item(),
                "bias": y_pred.item() - y_true[i].item(),
                "calibration": c,
                "ci_lower": ci_lower.item(),
                "ci_upper": ci_upper.item(),
                "ci_width": ci_upper.item() - ci_lower.item(),
                "ci_covered": ci_covered,
                "ci_lower_scaled": ci_lower_scaled.item(),
                "ci_upper_scaled": ci_upper_scaled.item(),
                "ci_width_scaled": ci_upper_scaled.item() - ci_lower_scaled.item(),
                "ci_covered_scaled": ci_lower_scaled.item() <= y_obs[i] <= ci_upper_scaled.item(),
                "pi_lower": pi_lower.item(),
                "pi_upper": pi_upper.item(),
                "pi_width": pi_upper.item() - pi_lower.item(),
                "pi_covered": pi_covered,
                "pi_lower_scaled": pi_lower_scaled.item(),
                "pi_upper_scaled": pi_upper_scaled.item(),
                "pi_width_scaled": pi_upper_scaled.item() - pi_lower_scaled.item(),
                "pi_covered_scaled": pi_lower_scaled.item() <= y_obs[i] <= pi_upper_scaled.item(),
                "ri_lower": ri_lower.item(),
                "ri_upper": ri_upper.item(),
                "ri_width": ri_upper.item() - ri_lower.item(),
                "ri_lower_scaled": ri_lower_scaled.item(),
                "ri_upper_scaled": ri_upper_scaled.item(),
                "ri_width_scaled": ri_upper_scaled.item() - ri_lower_scaled.item(),
                "cfml_lower": cfml_lower.item(),
                "cfml_upper": cfml_upper.item(),
                "cfml_width": cfml_upper.item() - cfml_lower.item(),
                "cfml_covered": cfml_covered,
                "rn_norm": (rn_norm.item() if type(rn_norm) == np.ndarray else rn_norm),
                "sigma_hat": sigma_hat.item(),
                "tau_hat": tau_hat.item(),
                "pi_sigma": pi_sigma.item()
            }

    # Dataframe setup
    with ProcessPoolExecutor(max_workers=6) as executor:
        result = tqdm(executor.map(training_rep, range(rep)), 
                      desc=f"Training BRATD (lr={lr}, dr={dr}, sr={sr}, max_depth={max_depth})", total=rep)
        rows = list(result)
        executor.shutdown()
        

    df = pd.DataFrame(rows)

    # ri coverage rate
    for k in range(rep):
        for i in range(num_test_pts):
            current_row = df[(df["model_idx"] == k) & (df["pt_idx"] == i)]
            if current_row.empty:
                continue

            ri_lower = current_row["ri_lower"].values[0]
            ri_upper = current_row["ri_upper"].values[0]
            ri_lower_scaled = current_row["ri_lower_scaled"].values[0]
            ri_upper_scaled = current_row["ri_upper_scaled"].values[0]

            other_models = df[(df["model_idx"] != k) & (df["pt_idx"] == i)]
            ri_cov_count = 0
            ri_scaled_cov_count = 0

            for _, row in other_models.iterrows():
                y_pred_other = row["y_pred"]
                if ri_lower <= y_pred_other <= ri_upper:
                    ri_cov_count += 1
                if ri_lower_scaled <= y_pred_other <= ri_upper_scaled:
                    ri_scaled_cov_count += 1

            df.loc[(df["model_idx"] == k) & (df["pt_idx"] == i), "ri_coverage"] = ri_cov_count / (rep - 1)
            df.loc[(df["model_idx"] == k) & (df["pt_idx"] == i), "ri_coverage_scaled"] = ri_scaled_cov_count / (rep - 1)

    # mean coverage rate
    mean_coverage = df.groupby("pt_idx").agg(
        ci_coverage_mean=("ci_covered", "mean"),
        ci_scaled_coverage_mean=("ci_covered_scaled", "mean"),
        pi_coverage_mean=("pi_covered", "mean"),
        pi_scaled_coverage_mean=("pi_covered_scaled", "mean"),
        ri_coverage_mean=("ri_coverage", "mean"),
        ri_scaled_coverage_mean=("ri_coverage_scaled", "mean"),
        conf_coverage_mean=("cfml_covered", "mean"),
    ).reset_index()


    # merge the mean coverage rate with the original dataframe
    df = df.merge(mean_coverage, on="pt_idx", how="left")

    # Save results
    if not os.path.exists('./reports/coverage_rates/'):
        os.makedirs('./reports/coverage_rates/')
    output_filename = f"npts_{num_test_pts}_rep_{rep}_epo_{epoch}_lr_{lr}_sr_{sr}_dr_{dr}_md_{max_depth}_nys_{Nystrom_subsample}_in_bag_{in_bag}"
    output_path = os.path.join(f"./reports/coverage_rates/{function}", f"{output_filename}.parquet")
    df.to_parquet(output_path)

    # Plotting
    ci_coverage = df[['pt_idx', 'ci_coverage_mean']].rename(columns={'ci_coverage_mean': 'coverage_rate'})
    ci_coverage['type'] = "Confidence Interval"

    pi_coverage = df[['pt_idx', 'pi_coverage_mean']].rename(columns={'pi_coverage_mean': 'coverage_rate'})
    pi_coverage['type'] = "Prediction Interval"

    ri_coverage = df[['pt_idx', 'ri_coverage_mean']].rename(columns={'ri_coverage_mean': 'coverage_rate'})
    ri_coverage['type'] = "Reproduction Interval"

    cfml_coverage = df[['pt_idx', 'conf_coverage_mean']].rename(columns={'conf_coverage_mean': 'coverage_rate'})
    cfml_coverage['type'] = "Conformal Interval"

    # Combine all
    all_df = pd.concat([ci_coverage, pi_coverage, ri_coverage, cfml_coverage], ignore_index=True)
    all_df['type'] = all_df['type'].astype('category')

    # Plot
    palette = {
        "Confidence Interval": "#00BEFF",
        "Prediction Interval": "#F8766D",
        "Reproduction Interval": "#7CAE00",
        "Conformal Interval": "#C77CFF"
    }

    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(8, 2), sharex=False)

    # plot each interval type on its own ax
    pt.RainCloud(
        x='type', y='coverage_rate', data=ci_coverage,
        palette=palette, bw=0.5, width_viol=1.2,
        ax=axes[0], orient='h'
    )
    axes[0].set_title("Confidence Interval")
    axes[0].set_xlabel("Coverage")

    pt.RainCloud(
        x='type', y='coverage_rate', data=pi_coverage,
        palette=palette, bw=0.5, width_viol=1.2,
        ax=axes[1], orient='h'
    )
    axes[1].set_title("Prediction Interval")
    axes[1].set_xlabel("Coverage")

    pt.RainCloud(
        x='type', y='coverage_rate', data=ri_coverage,
        palette=palette, bw=0.5, width_viol=1.2,
        ax=axes[2], orient='h'
    )
    axes[2].set_title("Reproduction Interval")
    axes[2].set_xlabel("Coverage")

    pt.RainCloud(
        x='type', y='coverage_rate', data=cfml_coverage,
        palette=palette, bw=0.5, width_viol=1.2,
        ax=axes[3], orient='h'
    )
    axes[3].set_title("Conformal Interval")
    axes[3].set_xlabel("Coverage")

    # common labels
    for ax in axes:
        ax.set_ylabel("")   # or ax.set_ylabel("coverage_rate") if you prefer

    plt.tight_layout()
    if not os.path.exists('./plots/coverage_rates/cov_rate'):
        os.makedirs('./plots/coverage_rates/cov_rate')
    plot_path = os.path.join(f"./plots/coverage_rate/cov_rate/{function}", f"{output_filename}.png")
    plt.savefig(plot_path)
    plt.close()

    # Calculate mean width for each interval type at each test point
    mean_width = df.groupby("pt_idx").agg(
        ci_width_mean=("ci_width", "mean"),
        pi_width_mean=("pi_width", "mean"),
        ri_width_mean=("ri_width", "mean"),
        cfml_width_mean=("cfml_width", "mean")
    ).reset_index()

    # Merge the mean widths back into the original dataframe
    df = df.merge(mean_width, on="pt_idx", how="left")

    # Prepare data for plotting average widths
    ci_width = df[['pt_idx', 'ci_width_mean']].rename(columns={'ci_width_mean': 'width'})
    ci_width['type'] = "Confidence Interval"

    pi_width = df[['pt_idx', 'pi_width_mean']].rename(columns={'pi_width_mean': 'width'})
    pi_width['type'] = "Prediction Interval"

    ri_width = df[['pt_idx', 'ri_width_mean']].rename(columns={'ri_width_mean': 'width'})
    ri_width['type'] = "Reproduction Interval"

    cfml_width = df[['pt_idx', 'cfml_width_mean']].rename(columns={'cfml_width_mean': 'width'})
    cfml_width['type'] = "Conformal Interval"

    # Combine all width data
    all_width_df = pd.concat([ci_width, pi_width, ri_width], ignore_index=True)
    all_width_df['type'] = all_width_df['type'].astype('category')

    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(8, 2), sharex=False)

    # 1) Confidence Interval widths
    pt.RainCloud(
        x='type', y='width', data=ci_width,
        palette=palette, bw=0.5, width_viol=1.2,
        ax=axes[0], orient='h'
    )
    axes[0].set_title("Confidence Interval Width")

    # 2) Prediction Interval widths
    pt.RainCloud(
        x='type', y='width', data=pi_width,
        palette=palette, bw=0.5, width_viol=1.2,
        ax=axes[1], orient='h'
    )
    axes[1].set_title("Prediction Interval Width")

    # 3) Reproduction Interval widths
    pt.RainCloud(
        x='type', y='width', data=ri_width,
        palette=palette, bw=0.5, width_viol=1.2,
        ax=axes[2], orient='h'
    )
    axes[2].set_title("Reproduction Interval Width")

    # 4) Conformal Interval widths
    pt.RainCloud(
        x='type', y='width', data=cfml_width,
        palette=palette, bw=0.5, width_viol=1.2,
        ax=axes[3], orient='h'
    )
    axes[3].set_title("Conformal Interval Width")

    # common x‐label on bottom plot
    axes[-1].set_xlabel("Average Width")
    # remove redundant y‐labels if you like
    for ax in axes:
        ax.set_xlabel("")

    plt.tight_layout()
    # Save Average Width Plot
    if not os.path.exists('./plots/coverage_rates/width'):
        os.makedirs('./plots/coverage_rates/width')
    average_width_plot_path = os.path.join(
        f"./plots/coverage_rate/width/{function}",
        f"{output_filename}_width.png"
    )
    plt.savefig(average_width_plot_path)
    plt.close()

print("Grid search complete. Results saved.")