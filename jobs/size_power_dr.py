#!/usr/bin/env python3

import sys, os
sys.path.insert(0, os.path.abspath("../src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from BRAT.algorithms import BRATD

# Set a fixed RNG for reproducibility
rng = np.random.default_rng(66)

def run_test(n, weight_third=0.0, d=5, m_hold_pct=0.5, **common):
    """
    Runs the variable importance test for one dataset of size n.
    weight_third: signal weight for the third covariate (null=0, alternative>0).
    Returns the p-value of the chi-squared test.
    """
    # Generate data
    X = rng.uniform(0, 5, (n, d))
    y = (4 * X[:, 0]
         - X[:, 1]**2
         + weight_third * X[:, 2]
         + rng.normal(scale=0.01, size=n)
    )

    # Train/hold split
    X_train, X_hold, y_train, y_hold = train_test_split(
        X, y, test_size=m_hold_pct, random_state=0
    )

    # Subset for null model
    d_check = 2
    X_train_sub = X_train[:, :d_check]
    X_hold_sub = X_hold[:, :d_check]

    # Fit full and sub models
    model_full = BRATD(**common)
    model_full.fit(X_train, y_train, X_hold, y_hold)
    model_sub = BRATD(**common)
    model_sub.fit(X_train_sub, y_train, X_hold_sub, y_hold)

    # Precompute kernel sketches
    model_full.full_K()
    model_sub.full_K()
    m_pts = X_hold.shape[0]
    n_train = X_train.shape[0]

    # Build R matrices
    R_full = np.zeros((m_pts, n_train))
    R_sub  = np.zeros((m_pts, n_train))
    for j, x in enumerate(X_hold):
        rn_full, _ = model_full.sketch_r(x, vector=True)
        rn_sub,  _ = model_sub.sketch_r(x[:d_check], vector=True)
        R_full[j, :] = rn_full
        R_sub[j, :] = rn_sub

    R_diff = R_full - R_sub

    # Estimate variance and compute test statistic
    sigma2 = model_full.est_sigma_hat2(in_bag=False)
    Sigma = sigma2 * (R_diff @ R_diff.T)
    Sigma_inv = np.linalg.pinv(Sigma)
    delta = model_full.predict(X_hold) - model_sub.predict(X_hold_sub)
    T = float(delta.T @ Sigma_inv @ delta)
    p_val = 1 - chi2.cdf(T, df=m_pts)
    return p_val


def main():
    # Simulation settings
    n = 100                # total sample size
    reps = 30              # number of repetitions per setting
    retaining_rates = np.arange(0.1, 1.01, 0.1)
    dropout_rates = 1.0 - retaining_rates

    # Store means and stds for error rates
    type1_means = []
    type1_stds  = []
    type2_means = []
    type2_stds  = []

    # Base parameters (dropout_rate is set per iteration)
    common_base = dict(
        n_estimators   = 100,
        learning_rate  = 0.8,
        max_depth      = 8,
        subsample_rate = 1.0,
        disable_tqdm   = True
    )

    # Loop over retaining rates
    for dr in retaining_rates:
        actual_dr = 1.0 - dr
        print(f"Running simulations for dropout_rate = {actual_dr:.1f}")
        common = {**common_base, 'dropout_rate': actual_dr}

        # Run replications
        pvals_null = [run_test(n, weight_third=0.0, **common) for _ in tqdm(range(reps), desc="Type I")]
        pvals_alt  = [run_test(n, weight_third=5.0, **common) for _ in tqdm(range(reps), desc="Type II")]

        # Compute empirical means and stds
        errs1 = (np.array(pvals_null) < 0.05).astype(float)
        errs2 = (np.array(pvals_alt)  < 0.05).astype(float)
        type1_means.append(errs1.mean())
        type1_stds .append(errs1.std(ddof=0))
        type2_means.append(1 - errs2.mean())
        type2_stds .append((1 - errs2).std(ddof=0))

        print(f"dropout_rate={actual_dr:.1f}, Type I error={type1_means[-1]:.3f} ± {type1_stds[-1]:.3f}, "
              f"Type II error={type2_means[-1]:.3f} ± {type2_stds[-1]:.3f}")

    # Save summary to CSV
    results = pd.DataFrame({
        'dropout_rate': dropout_rates,
        'type_I_mean':   type1_means,
        'type_I_std':    type1_stds,
        'type_II_mean':  type2_means,
        'type_II_std':   type2_stds,
    })
    out_dir = './variable_importance/'
    os.makedirs(out_dir, exist_ok=True)
    file_path = os.path.join(out_dir, 'size_power_dr.csv')
    results.to_csv(file_path, index=False)
    print(f"Saved summary to {file_path}")

if __name__ == "__main__":
    main()
