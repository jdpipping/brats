import numpy as np
import pandas as pd
import os
from scipy.stats import norm
from tqdm import tqdm
from itertools import product
from sklearn.model_selection import ParameterSampler
from BRAT.brat import BRATD
from BRAT.utils import generate_data
import matplotlib.pyplot as plt
plt.style.use('/Users/longbo/Desktop/Boulevard/Simulations/DropBoostForests/matplotlibrc')
plt.rcParams['xtick.labelsize'] = 6
plt.rcParams['ytick.labelsize'] = 6

import matplotlib.colors as mcolors

def darken(color, factor=0.7):
    """
    Darken a hex color by multiplying its RGB components by `factor` (0 < factor < 1).
    """
    rgb = np.array(mcolors.to_rgb(color))
    darker = np.clip(rgb * factor, 0, 1)
    return mcolors.to_hex(darker)


np.random.seed(42)
n_train = 1000
n_test = 300
n_cal = 200

"""
ne_list = [200, 300]
lr_list = [0.3, 1.0]
md_list = [8, 16]
sr_list = [0.2, 0.8]
dr_list = [0.3, 0.9]
"""
"""
ne_list = [500]
lr_list = [0.6, 1.0]
md_list = [8, 12, 16]
sr_list = [0.3, 0.5, 0.8]
dr_list = [0.5, 0.9]
"""

# 1) define your parameter distributions
param_dist = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.3, 0.6, 1.0],
    'max_depth':      [4,  8,  12],
    'subsample_rate': [0.3, 0.6, 0.9],
    'dropout_rate':   [0.3, 0.6, 0.9]
}

# 2) sample n_iter random combinations
n_iter = 30   # << your budget
sampler = list(ParameterSampler(param_dist, n_iter=n_iter, random_state=42))

noise_std = 1

# 
f = lambda x: np.sin(2*np.pi*x) + 0.5 * x**2

# uniform samples on [0,1]
X_train = np.random.rand(n_train,1)   # shape (n_train,1)
X_test  = np.random.rand(n_test,1)   # shape (n_test,1)
X_cal   = np.random.rand(n_cal,1)   # shape (n_cal,1)

# keep smooth grid for plotting
X_plot  = np.linspace(0, 1, n_test).reshape(-1,1)

# true signal
y_train_true = f(X_train).flatten()
y_test_true  = f(X_test).flatten()
y_cal_true   = f(X_cal).flatten()

# add noise
y_train = y_train_true + noise_std * np.random.randn(n_train)
y_test  = y_test_true  + noise_std * np.random.randn(n_test)
y_cal   = y_cal_true + noise_std * np.random.randn(n_cal)

results = []

reps = 50

output_dir = '/Users/longbo/Desktop/Boulevard/Simulations/DropBoostForests/reports/1d_intervals_df'
os.makedirs(output_dir, exist_ok=True)

for params in tqdm(sampler,
                               desc='Grid Searching...'):
    # extract and cast to the right types
    ne = int(params['n_estimators'])
    lr = float(params['learning_rate'])
    md = int(params['max_depth'])
    sr = float(params['subsample_rate'])
    dr = float(params['dropout_rate'])

    model_results = []
    for model_id in range(reps):
        # Train BRATD model
        bratd = BRATD(n_estimators=ne,
                    learning_rate=lr,
                    max_depth=md,
                    subsample_rate=sr,
                    dropout_rate=dr,
                    disable_tqdm=True)
        bratd.fit(X_train, y_train, X_test, y_test)

        # Predict and compute CIs for each test point
        bratd_pred = []
        bratd_ci_lb = []
        bratd_ci_ub = []
        bratd_pi_lb = []
        bratd_pi_ub = []
        bratd_ri_lb = []
        bratd_ri_ub = []
        counter = 0

        # Precompute the K matrix
        sigma_hat2 = bratd.est_sigma_hat2(in_bag=False)
        bratd.unif_nystrom(Nystrom_subsample=0.05)
        bratd.sketch_K()

        y_pred_cal = bratd.predict(X_cal)
        res_cal = np.abs(y_pred_cal - y_cal)
        qtel_hat = np.quantile(res_cal, 0.95)

        for i, x in tqdm(enumerate(X_test), desc='Iterating test points...'):
            pred = bratd.predict(x.reshape(1, -1))
            bratd_pred.append(pred)
            rn_norm = bratd.sketch_r(x)
            s = (1 + bratd.learning_rate * (1 - bratd.dropout_rate)) / bratd.learning_rate
            tau_hat2 = (s**2 * rn_norm**2 * sigma_hat2)

            ci_lb = pred - norm.ppf(0.975) * np.sqrt(tau_hat2)
            ci_ub = pred + norm.ppf(0.975) * np.sqrt(tau_hat2)
            pi_lb = pred - norm.ppf(0.975) * np.sqrt(tau_hat2 + sigma_hat2)
            pi_ub = pred + norm.ppf(0.975) * np.sqrt(tau_hat2 + sigma_hat2)
            ri_lb = pred - norm.ppf(0.975) * np.sqrt(2 * tau_hat2)
            ri_ub = pred + norm.ppf(0.975) * np.sqrt(2 * tau_hat2)
            cfml_lower = pred - qtel_hat
            cfml_upper = pred + qtel_hat

            ci_covered = int(ci_lb.item() <= y_test_true[i].item() <= ci_ub.item())
            cfml_covered = int(cfml_lower.item() <= y_test[i].item() <= cfml_upper.item())
            pi_covered = int(pi_lb.item() <= y_test[i].item() <= pi_ub.item())

            model_results.append({
                'pt_idx': i,
                'model_id': model_id,
                'pred': pred.item(),
                'obs': y_test[i].item(),
                'truth': y_test_true[i],
                'bias': pred.item() - y_test_true[i],
                'tau_hat2': tau_hat2.item(),
                'sigma_hat2': sigma_hat2.item(),
                'rn_norm': rn_norm,
                'ci_lb': ci_lb.item(),
                'ci_ub': ci_ub.item(),
                'ci_width': ci_ub.item() - ci_lb.item(),
                'ci_covered': ci_covered,
                'pi_lb': pi_lb.item(),
                'pi_ub': pi_ub.item(),
                'pi_width': pi_ub.item() - pi_lb.item(),
                'pi_covered': pi_covered,
                'ri_lb': ri_lb.item(),
                'ri_ub': ri_ub.item(),
                'ri_width': ri_ub.item() - ri_lb.item(),
                'cfml_lower': cfml_lower.item(),
                'cfml_upper': cfml_upper.item(),
                'cfml_width': cfml_upper.item() - cfml_lower.item(),
                'cfml_covered': cfml_covered,
            })
    
    df = pd.DataFrame(model_results)
    df['ci_mean_lb'] = df.groupby('pt_idx')['ci_lb'].transform('mean')
    df['ci_mean_ub'] = df.groupby('pt_idx')['ci_ub'].transform('mean')
    df['pi_mean_lb'] = df.groupby('pt_idx')['pi_lb'].transform('mean')
    df['pi_mean_ub'] = df.groupby('pt_idx')['pi_ub'].transform('mean')
    df['ri_mean_lb'] = df.groupby('pt_idx')['ri_lb'].transform('mean')
    df['ri_mean_ub'] = df.groupby('pt_idx')['ri_ub'].transform('mean')
    df['cfml_mean_lb'] = df.groupby('pt_idx')['cfml_lower'].transform('mean')
    df['cfml_mean_ub'] = df.groupby('pt_idx')['cfml_upper'].transform('mean')

    ci_coverage_rate = df.groupby('pt_idx')['ci_covered'].mean()
    df['ci_coverage'] = ci_coverage_rate
    mean_ci_coverage = df['ci_coverage'].mean()
    df['mean_ci_coverage'] = mean_ci_coverage

    cfml_coverage_rate = df.groupby('pt_idx')['cfml_covered'].mean()
    df['cfml_coverage'] = cfml_coverage_rate
    mean_cfml_coverage = df['cfml_coverage'].mean()
    df['mean_cfml_coverage'] = mean_cfml_coverage

    pi_coverage_rate = df.groupby('pt_idx')['pi_covered'].mean()
    df['pi_coverage'] = pi_coverage_rate
    mean_pi_coverage = df['pi_coverage'].mean()
    df['mean_pi_coverage'] = mean_pi_coverage

    ri_coverage_rate = []
    for _, row in df.iterrows():
        ri_coverage_rate.append(
            ((df['pred'] >= row['ri_lb']) & (df['pred'] <= row['ri_ub'])).mean()
        )

    df['ri_coverage'] = ri_coverage_rate
    mean_ri_coverage = df['ri_coverage'].mean()
    df['mean_ri_coverage'] = mean_ri_coverage
    results.append(df)

    file_name = f"ne_{ne}_lr_{lr}_md_{md}_sr_{sr}_dr_{dr}.csv"
    df.to_csv(os.path.join(output_dir, file_name), index=False)

    bratd_mean = df.groupby('pt_idx')['pred'].mean().values
    bratd_mean_ci_lb = df.groupby('pt_idx')['ci_lb'].mean().values
    bratd_mean_ci_ub = df.groupby('pt_idx')['ci_ub'].mean().values
    bratd_mean_pi_lb = df.groupby('pt_idx')['pi_lb'].mean().values
    bratd_mean_pi_ub = df.groupby('pt_idx')['pi_ub'].mean().values
    bratd_mean_ri_lb = df.groupby('pt_idx')['ri_lb'].mean().values
    bratd_mean_ri_ub = df.groupby('pt_idx')['ri_ub'].mean().values
    bratd_mean_cfml_lb = df.groupby('pt_idx')['cfml_lower'].mean().values
    bratd_mean_cfml_ub = df.groupby('pt_idx')['cfml_upper'].mean().values
    
    sort_idx = np.argsort(X_test.ravel())
    X_test_s = X_test.ravel()[sort_idx]
    pred_s = bratd_mean[sort_idx]
    ci_lb_s = bratd_mean_ci_lb[sort_idx]
    ci_ub_s = bratd_mean_ci_ub[sort_idx]
    pi_lb_s = bratd_mean_pi_lb[sort_idx]
    pi_ub_s = bratd_mean_pi_ub[sort_idx]
    ri_lb_s = bratd_mean_ri_lb[sort_idx]
    ri_ub_s = bratd_mean_ri_ub[sort_idx]
    cfml_lb_s = bratd_mean_cfml_lb[sort_idx]
    cfml_ub_s = bratd_mean_cfml_ub[sort_idx]

    data_dir = '/Users/longbo/Desktop/Boulevard/Simulations/DropBoostForests/plots/1d_intervals/data'
    os.makedirs(data_dir, exist_ok=True)

    test_df = pd.DataFrame({
        'X_test':    X_test_s,
        'pred':      pred_s,
        'ci_lb':     ci_lb_s,
        'ci_ub':     ci_ub_s,
        'pi_lb':     pi_lb_s,
        'pi_ub':     pi_ub_s,
        'ri_lb':     ri_lb_s,
        'ri_ub':     ri_ub_s,
        'cfml_lb':   cfml_lb_s,
        'cfml_ub':   cfml_ub_s,
        'mean_ci_coverage':  mean_ci_coverage,
        'mean_pi_coverage':  mean_pi_coverage,
        'mean_ri_coverage':  mean_ri_coverage,
        'mean_cfml_coverage':mean_cfml_coverage
    })

    file_tag = f'ne_{ne}_lr_{lr}_md_{md}_sr_{sr}_dr_{dr}.csv'
    test_df.to_csv(os.path.join(data_dir, file_tag), index=False)

    # Plotting all four plots side by side
    fig, axes = plt.subplots(1, 4, figsize=(8, 2), sharey=True)

    true_plot = f(X_plot)

    # 1) Confidence Interval
    ax = axes[0]
    ax.scatter(X_train, y_train, color='grey', alpha=0.3, s=5, label='Training Data')
    ax.plot(X_test_s, pred_s, '-', color=darken("#00BEFF",1.0), linewidth=1, label='Mean Prediction')
    ax.fill_between(X_test_s, ci_lb_s, ci_ub_s, facecolor="#00BEFF", edgecolor="#00BEFF", alpha=0.4)
    ax.plot(X_plot, true_plot, 'k--', linewidth=1, zorder=5, label='True Function')  # true on top
    ax.text(0.05, 0.95, f'Coverage: {mean_ci_coverage:.2%}',
            transform=ax.transAxes, fontsize=8, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.2))
    leg = ax.legend(['True Function', 'Training Data', 'Mean Prediction', 'CI'], loc='lower left', fontsize=6)
    leg.get_frame().set_alpha(0.3)
    ax.set_title("CI", fontsize=9)

    # 2) Reproduction Interval  ← swapped into position 2
    ax = axes[1]
    ax.scatter(X_train, y_train, color='grey', alpha=0.3, s=5)
    ax.plot(X_test_s, pred_s, '-', color=darken("#7CAE00",1.0), linewidth=1)
    ax.fill_between(X_test_s, ri_lb_s, ri_ub_s, facecolor="#7CAE00", edgecolor="#7CAE00", alpha=0.4)
    ax.plot(X_plot, true_plot, 'k--', linewidth=1, zorder=5, label='True Function')
    ax.text(0.05, 0.95, f'Coverage: {mean_ri_coverage:.2%}',
            transform=ax.transAxes, fontsize=8, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.2))
    leg = ax.legend(['True Function', 'Training Data', 'Mean Prediction', 'RI'], loc='lower left', fontsize=6)  # adjust legend
    leg.get_frame().set_alpha(0.3)
    ax.set_title("RI", fontsize=9)

    # 3) Prediction Interval  ← swapped into position 3
    ax = axes[2]
    ax.scatter(X_train, y_train, color='grey', alpha=0.3, s=5)
    ax.plot(X_test_s, pred_s, '-', color=darken("#F8766D",1.0), linewidth=1)
    ax.fill_between(X_test_s, pi_lb_s, pi_ub_s, facecolor="#F8766D", edgecolor="#F8766D", alpha=0.4)
    ax.plot(X_plot, true_plot, 'k--', linewidth=1, zorder=5, label='True Function')
    ax.text(0.05, 0.95, f'Coverage: {mean_pi_coverage:.2%}',
            transform=ax.transAxes, fontsize=8, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.2))
    leg = ax.legend(['True Function', 'Training Data', 'Mean Prediction', 'PI'], loc='lower left', fontsize=6)
    leg.get_frame().set_alpha(0.3)
    ax.set_title("PI", fontsize=9)

    # 4) Conformal Interval
    ax = axes[3]
    ax.scatter(X_train, y_train, color='grey', alpha=0.3, s=5)
    ax.plot(X_test_s, pred_s, '-', color=darken("#C77CFF",1.0), linewidth=1)
    ax.fill_between(X_test_s, cfml_lb_s, cfml_ub_s, facecolor="#C77CFF", edgecolor="#C77CFF", alpha=0.4)
    ax.plot(X_plot, true_plot, 'k--', linewidth=1, zorder=5, label='True Function')
    ax.text(0.05, 0.95, f'Coverage: {mean_cfml_coverage:.2%}',
            transform=ax.transAxes, fontsize=8, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.2))
    leg = ax.legend(['True Function', 'Training Data', 'Mean Prediction', 'Conformal PI'], loc='lower left', fontsize=6)
    leg.get_frame().set_alpha(0.3)
    ax.set_title("Conformal PI", fontsize=9)

    plt.tight_layout()
    path = '/Users/longbo/Desktop/Boulevard/Simulations/DropBoostForests/plots/1d_intervals/'
    fig.savefig(path + f'ne_{ne}_lr_{lr}_md_{md}_sr_{sr}_dr_{dr}.png', dpi=500)
    plt.close(fig)