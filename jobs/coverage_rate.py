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
import itertools
import ptitprince as pt
from scipy.stats import norm
from concurrent.futures import ProcessPoolExecutor

plt.style.use('matplotlibrc')

# Hyperparameter grid

lr_values = [0.6] #[0.3, 0.6, 1.0]
dr_values = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]#[0, 0.3, 0.6, 0.9]
sr_values = [0.8]
max_depth_values = [4] #[4,6,8]


# lr_values = [0.6]
# dr_values = [0.3]
# sr_values = [0.8]
# max_depth_values = [4]

# Fixed settings
num_test_pts = 200
rep = 30
epoch = 200
Nystrom_subsample = 0.1
in_bag = False
function = 'friedman1'
conformal_adjustment = True

# Create directories for saving results
os.makedirs("../reports", exist_ok=True)
os.makedirs("../plots/coverage_rate", exist_ok=True)




# Iterate over all combinations of hyperparameters
for lr, dr, sr, max_depth in product(lr_values, dr_values, sr_values, max_depth_values):
    

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

        print(str(k)+':', 'fitting model')
        bratd.fit(X_train, y_train, X_test, y_test)
        _, _, _ = bratd.unif_nystrom(Nystrom_subsample)
        bratd.sketch_K()
        sigma_hat2 = bratd.est_sigma_hat2(in_bag)
        sigma_hat = np.sqrt(sigma_hat2)
        lam = bratd.learning_rate
        q = 1 - bratd.dropout_rate
        s = (1 + lam * q) / lam

        y_pred_cal = bratd.predict(X_cal)
        res_cal = np.abs(y_pred_cal - y_cal)
        qtle_hat = np.quantile(res_cal, 1 - 0.05)

        
        if conformal_adjustment:
            print(str(k)+':', 'computing PI inflation factor')
            pi_sigma_cals = []
            for i, x in tqdm(enumerate(X_cal), 
                          desc=f"Repetition {k} calibration: "):
                rn_norm_cal = bratd.sketch_r(x.reshape(1,-1))
                tau_hat_cal = s * rn_norm_cal * sigma_hat
                pi_sigma_cal = np.sqrt(sigma_hat2 + tau_hat_cal**2)
                pi_sigma_cals.append(pi_sigma_cal)
            pi_sigma_cal = np.array(pi_sigma_cal)
            c_pi = find_min_scale(
                    y_true=y_cal,            # for PI you compare to noisy obs
                    y_pred=y_pred_cal,
                    widths=norm.ppf(0.975) * pi_sigma_cal,
                    target=0.95
                )
        else:
            c_pi = 1.96

        print(str(k)+':', 'computing coverages')
        rows = []
        for i, x in tqdm(enumerate(test_points), 
                      desc=f"Repetition {k} coverage: "):
            y_pred = bratd.predict(x.reshape(1, -1))
            rn_norm = bratd.sketch_r(x.reshape(1, -1))
            tau_hat = s * rn_norm * sigma_hat
            pi_sigma = np.sqrt(sigma_hat2 + tau_hat**2)

            ci_lower = y_pred - norm.ppf(0.975) * c_pi * tau_hat
            ci_upper = y_pred + norm.ppf(0.975) * c_pi * tau_hat
            ci_covered = int(ci_lower.item() <= y_true[i] <= ci_upper.item())


            pi_lower = y_pred - norm.ppf(0.975) * c_pi * pi_sigma
            pi_upper = y_pred + norm.ppf(0.975) * c_pi * pi_sigma
            ri_lower = y_pred - norm.ppf(0.975) * c_pi * np.sqrt(2) * tau_hat
            ri_upper = y_pred + norm.ppf(0.975) * c_pi * np.sqrt(2) * tau_hat

            cfml_lower = y_pred - qtle_hat
            cfml_upper = y_pred + qtle_hat
            cfml_covered = int(cfml_lower.item() <= y_obs[i] <= cfml_upper.item())
            pi_covered = int(pi_lower.item() <= y_obs[i] <= pi_upper.item())
            
            rows.append({
                "model_idx": k,
                "pt_idx": i,
                "y_true": y_true[i].item(),
                "y_obs": y_obs[i].item(),
                "y_pred": y_pred.item(),
                "bias": y_pred.item() - y_true[i].item(),
                "ci_lower": ci_lower.item(),
                "ci_upper": ci_upper.item(),
                "ci_width": ci_upper.item() - ci_lower.item(),
                "ci_covered": ci_covered,
                "pi_lower": pi_lower.item(),
                "pi_upper": pi_upper.item(),
                "pi_width": pi_upper.item() - pi_lower.item(),
                "pi_covered": pi_covered,
                "ri_lower": ri_lower.item(),
                "ri_upper": ri_upper.item(),
                "ri_width": ri_upper.item() - ri_lower.item(),
                "cfml_lower": cfml_lower.item(),
                "cfml_upper": cfml_upper.item(),
                "cfml_width": cfml_upper.item() - cfml_lower.item(),
                "cfml_covered": cfml_covered,
                "rn_norm": (rn_norm.item() if type(rn_norm) == np.ndarray else rn_norm),
                "sigma_hat": sigma_hat.item(),
                "tau_hat": tau_hat.item(),
                "pi_sigma": pi_sigma.item()
            })
        return rows

    # Dataframe setup
    with ProcessPoolExecutor(max_workers=31) as executor:
        result = tqdm(executor.map(training_rep, range(rep)), 
                      desc=f"Training BRATD (lr={lr}, dr={dr}, sr={sr}, max_depth={max_depth})", total=rep)
        rows = list(result)
        executor.shutdown()

    print(f"Concatenating dataframe for  (lr={lr}, dr={dr}, sr={sr}, max_depth={max_depth})")
    df = pd.DataFrame(list(itertools.chain.from_iterable(rows)))

    # pi and ri coverage rate
    for k in range(rep):
        for i in range(num_test_pts):
            current_row = df[(df["model_idx"] == k) & (df["pt_idx"] == i)]
            if current_row.empty:
                continue

            ri_lower = current_row["ri_lower"].values[0]
            ri_upper = current_row["ri_upper"].values[0]

            other_models = df[(df["model_idx"] != k) & (df["pt_idx"] == i)]
            ri_cov_count = 0

            for _, row in other_models.iterrows():
                y_pred_other = row["y_pred"]
                if ri_lower <= y_pred_other <= ri_upper:
                    ri_cov_count += 1

            df.loc[(df["model_idx"] == k) & (df["pt_idx"] == i), "ri_coverage"] = ri_cov_count / (rep - 1)

    # mean coverage rate
    mean_coverage = df.groupby("pt_idx").agg(
        ci_coverage_mean=("ci_covered", "mean"),
        pi_coverage_mean=("pi_covered", "mean"),
        ri_coverage_mean=("ri_coverage", "mean"),
        conf_coverage_mean=("cfml_covered", "mean"),
    ).reset_index()

    # merge the mean coverage rate with the original dataframe
    df = df.merge(mean_coverage, on="pt_idx", how="left")

    # Save results
    if not conformal_adjustment:
        if not os.path.exists('./reports/coverage_rates_conformal_False/'):
            os.makedirs('./reports/coverage_rates_conformal_False/')
        output_filename = f"npts_{num_test_pts}_rep_{rep}_epo_{epoch}_lr_{lr}_sr_{sr}_dr_{dr}_md_{max_depth}_nys_{Nystrom_subsample}_in_bag_{in_bag}"
        output_path = os.path.join(f"./reports/coverage_rates_conformal_False/{function}", f"{output_filename}.parquet")
        df.to_parquet(output_path)
    else:
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
    if conformal_adjustment:
        average_width_plot_path = os.path.join(
            f"./plots/coverage_rate/width/{function}",
            f"{output_filename}_width.png"
        )
    else:
        average_width_plot_path = os.path.join(
            f"./plots/coverage_rate/width/{function}_conformal_False",
            f"{output_filename}_width.png"
        )
    plt.savefig(average_width_plot_path)
    plt.close()

print("Grid search complete. Results saved.")