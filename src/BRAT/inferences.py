import numpy as np
import pandas as pd

from scipy.stats import norm

from BRAT.algorithms import BRATD, BRATP
from BRAT.variance_estimation import compute_k_vector, find_K_matrix, estimate_noise_variance, estimate_built_in_variance, calculate_rn
from BRAT.utils import generate_data
from tqdm import tqdm

def ground_truth(x, function_type='friedman1'):
    """
    Compute the ground truth function value for a given input x.
    
    Parameters:
      x: A numpy array representing the input features.
      function_type: The type of function to compute (default is 'friedman1').
      
    Returns:
      f_x: The computed function value.
    """
    
    if function_type == 'friedman1':
        f = lambda x: 10 * np.sin(np.pi * x[0] * x[1]) + 20 * (x[2] - 0.5)**2 + 10 * x[3] + 5 * x[4]
    elif function_type == 'friedman2':
        f = lambda x: np.sqrt(x[0]**2 + (x[1]*x[2] - 1/(x[1]*x[3] + 1e-6))**2)
    elif function_type == 'radial':
        f = lambda x: np.exp(-np.sum((x - 0.5)**2, dim=1))
    elif function_type == 'smooth_linear':
        f = lambda x: 3 * x[0] + 2 * x[1] - x[2] + 0.5 * np.sin(2 * np.pi * x[3]) + 0.3 * x[4]**2
    else:
        raise ValueError("Unknown function type. Choose 'friedman1', 'friedman2', or 'radial'.")
    
    y = f(x)
    
    return y
    

def simulated_hypothesis_test(BRAT_model, in_bag, x, f0, Nystrom_subsample=None):
    """
    Perform a hypothesis test to check if the predicted value at x is significantly different from f0.
    Parameters:
      BRAT_model: A trained BRAT model.
      in_bag: Boolean indicating whether to use in-bag samples for noise variance estimation.
      x: A test point as a numpy array (shape: (n_features,)).
      f0: The null hypothesis value to test against.
      Nystrom_subsample_rate: The subsample rate used to construct the kernel matrix.
    Returns:
      T: The test statistic.
      p_value: The p-value for the hypothesis test.
      y_pred: The predicted value at x.
      r_norm: The estimated L2 norm of the influence vector.
      sigma_hat: The estimated noise standard deviation.
      tau_hat: The estimated standard deviation of the prediction.
    """
    y_pred = BRAT_model.predict(np.expand_dims(x, axis=0))

    sigma_hat2, rn_norm, tau_hat2 = BRAT_model.est_tau_hat2(in_bag, Nystrom_subsample, x)
    tau_hat = np.sqrt(tau_hat2)
    sigma_hat = np.sqrt(sigma_hat2)
    
    T = (y_pred - f0) / (tau_hat)
    
    p_value = 2 * (1 - norm.cdf(np.abs(T)))
    
    return T, p_value, y_pred, rn_norm, sigma_hat, tau_hat

def PI(BRAT_model, in_bag, x, Nystrom_subsample=None, alpha=0.05):
    """
    Compute a 100(1-alpha)% prediction interval for the true function value f(x)
    using the BRAT model's built-in uncertainty quantification.

    The prediction interval is given by:
      [y_pred - z * sqrt(sigma2_hat + tau2_hat),
       y_pred + z * sqrt(sigma2_hat + tau2_hat)],
    where:
      - y_pred is the prediction for x,
      - sigma2_hat is the estimated noise standard deviation,
      - tau2_hat is a built-in estimate of the between-replication variance.

    Parameters:
      BRAT_model: A trained BRAT model.
      in_bag: Boolean indicating whether to use in-bag samples for noise variance estimation.
      x: A test point as a numpy array (shape: (n_features,)).
      Nystrom_subsample: The subsample rate for constructing kernel matrix
      alpha: Significance level (default 0.05).

    Returns:
      pi: A tuple (lower, upper) representing the prediction interval for f(x).
      y_pred: The predicted value at x.
      r_norm: The estimated L2 norm of the influence vector.
      sigma_hat2: The estimated noise variance.
      tau_hat2: The estimated between-replication variance.
    """
    y_pred = BRAT_model.predict(np.expand_dims(x, axis=0))[0]
    
    sigma_hat2, rn_norm, tau_hat2 = BRAT_model.est_tau_hat2(in_bag, Nystrom_subsample, x)
    
    pi_se = np.sqrt(sigma_hat2 + tau_hat2)
    
    z = norm.ppf(1 - alpha/2)
    
    lower = y_pred - z * pi_se
    upper = y_pred + z * pi_se
    
    return (lower, upper), y_pred, rn_norm, sigma_hat2, tau_hat2

def CI(BRAT_model, in_bag, x, Nystrom_subsample = None, alpha=0.05):
    """
    Compute a 100(1-alpha)% confidence interval for the true function value f(x)
    using the BRAT model's built-in uncertainty quantification.
    
    Under the CLT, we have:
      (s * hat_f(x) - f(x)) / (||r_n|| * sigma_epsilon) ~ N(0, 1)
    where s = lambda/(1+lambda*q).
    
    Thus, a 100(1-alpha)% CI for f(x) is given by:
      [s * hat_f(x) - z * (||r_n|| * sigma_hat),
       s * hat_f(x) + z * (||r_n|| * sigma_hat)],
    where z = norm.ppf(1-alpha/2).
    
    Parameters:
      BRAT_model: A trained BRAT model.
      in_bag: Boolean indicating whether to use in-bag samples for noise variance estimation.
      x: A test point as a numpy array (shape: (n_features,)).
      Nystrom_subsample: The subsample rate used to construct the kernel matrix.
      alpha: Significance level (default 0.05).
      
    Returns:
      ci: A tuple (lower, upper) representing the confidence interval for f(x).
      y_pred: The predicted value at x.
      rn_norm: The estimated L2 norm of the influence vector.
      sigma_hat2: The estimated noise variance.
      tau_hat2: The estimated variance of the prediction.
    """

    y_pred = BRAT_model.predict(x.reshape(1,-1))
    
    sigma_hat2, rn_norm, tau_hat2 = BRAT_model.est_tau_hat2(in_bag, Nystrom_subsample, x)

    tau_hat = np.sqrt(tau_hat2)
    z = norm.ppf(1 - alpha/2)

    lower = y_pred - z * (tau_hat)
    upper = y_pred + z * (tau_hat)
    
    return (lower, upper), y_pred, rn_norm, sigma_hat2, tau_hat2

def RI(BRAT_model, x, in_bag=False, Nystrom_subsample = None, alpha=0.05):
    """
    Compute a 100(1-alpha)% reproduction (prediction) interval for the true function value f(x)
    using the BRAT model's built-in uncertainty quantification, extended with a replication variance term.
    
    The reproduction interval is given by:
      [y_pred - z * sqrt(2 * tau2_hat),
       y_pred + z * sqrt(2 * tau2_hat)],
    where:
      - y_pred is the prediction for x,
      - tau2_hat is an estimate of the between-replication variance.

    Parameters:
      BRAT_model: A trained BRAT model.
      x: A test point as a numpy array (shape: (n_features,)).
      in_bag: Boolean indicating whether to use in-bag samples for noise variance estimation.
      Nystrom_subsample: The subsample rate used to construct the kernel matrix.
      alpha: Significance level (default 0.05).
      
    Returns:
      reproduction_interval: A tuple (lower, upper) representing the reproduction interval for f(x).
      y_pred: The predicted value at x.
      r_norm: The estimated L2 norm of the influence vector.
      sigma2_hat: The estimated noise variance.
      tau2_hat: The estimated between-replication variance.
    """
    
    y_pred = BRAT_model.predict(np.expand_dims(x, axis=0))[0]

    sigma_hat2, rn_norm, tau_hat2 = BRAT_model.est_tau_hat2(in_bag, Nystrom_subsample, x)

    ri_se = np.sqrt(2*tau_hat2)
    
    z = norm.ppf(1 - alpha/2)
    
    lower = y_pred - z * ri_se
    upper = y_pred + z * ri_se
    
    return (lower, upper), y_pred, rn_norm, sigma_hat2, tau_hat2

def all_intervals(BRAT_model, x, in_bag=False, Nystrom_subsample = None, alpha=0.05):
    """
    To avoid computation redundancy, this function returns the prediction intervals, confidence interval and reproduction interval of a BRAT model at a
    point of interest at the same time.
    The prediction interval is given by:
      [y_pred - z * sqrt((1+lam*q)^2/lam^2 * sigma2_hat + tau2_hat),
       y_pred + z * sqrt((1+lam*q)^2/lam^2 * sigma2_hat + tau2_hat)],
    where:
      - y_pred is the prediction for x,
      - sigma2_hat is the estimated noise variance,
      - tau2_hat is an estimate of the between-replication variance.
    The confidence interval is given by:
      [y_pred - z * tau_hat,
       y_pred + z * tau_hat],
    The reproduction interval is given by:
      [y_pred - z * sqrt(2 * tau2_hat),
       y_pred + z * sqrt(2 * tau2_hat)],
    where:
      - y_pred is the prediction for x,
      - sigma2_hat is the estimated noise variance,
      - tau2_hat is an estimate of the between-replication variance.

    Parameters:
      BRAT_model: A trained BRAT model.
      x: A test point as a numpy array (shape: (n_features,)).
      in_bag: Boolean indicating whether to use in-bag samples for noise variance estimation.
      Nystrom_subsample: The subsample rate used to construct the kernel matrix.
      alpha: Significance level (default 0.05).
    
    Returns:
      pi: A tuple (lower, upper) representing the prediction interval for f(x).
      ci: A tuple (lower, upper) representing the confidence interval for f(x).
      ri: A tuple (lower, upper) representing the reproduction interval for f(x).
      y_pred: The predicted value at x.
      r_norm: The estimated L2 norm of the influence vector.
      sigma2_hat: The estimated noise variance.
      tau2_hat: The estimated between-replication variance.
    """
    y_pred = BRAT_model.predict(np.expand_dims(x, axis=0))[0]

    sigma_hat2, rn_norm, tau_hat2 = BRAT_model.est_tau_hat2(in_bag, Nystrom_subsample, x)

    lam = BRAT_model.learning_rate
    q = 1 - BRAT_model.dropout_rate
    pi_se = np.sqrt((1+lam*q)**2/lam**2 * sigma_hat2 + tau_hat2)
    ci_se = np.sqrt(tau_hat2)
    ri_se = np.sqrt(2 * tau_hat2)

    z = norm.ppf(1 - alpha/2)

    pi_lower = y_pred - z * pi_se
    pi_upper = y_pred + z * pi_se

    ci_lower = y_pred - z * ci_se
    ci_upper = y_pred + z * ci_se

    ri_lower = y_pred - z * ri_se
    ri_upper = y_pred + z * ri_se

    return (pi_lower, pi_upper), (ci_lower, ci_upper), (ri_lower, ri_upper), y_pred, rn_norm, sigma2_hat, tau2_hat

def CI_coverage_rate(BRAT_model, in_bag, test_points, Nystrom_subsample, disable_tqdm, alpha = 0.05, ):
    """
    For a set of test points, compute and return whether each of them is covered by the CI given by a same BRAT model.
    Parameters:
      BRAT_model: A trained BRAT model.
      in_bag: Boolean indicating whether to use in-bag samples for noise variance estimation.
      test_points: The set of test points that are interested in.
      Nystrom_subsample: The subsample rate used to construct the kernel matrix.
      disable_tqdm: Boolean to disable the tqdm progress bar for individual trees(default True).
      alpha: Significance level for the reproduction interval (default 0.05).
    Returns:
        avg_coverage_rate: Float, average CI coverage over all replications.
        df: DataFrame with one row per BRAT replication:
            - 'ci_lower', 'ci_upper'
            - 'sigma2_hat', 'tau2_hat'
            - 'covered': whether CI covered the ground truth
            - 'truth': true f(x)
            - 'y_pred': model prediction at x
    """

    records = []
    _, _, _ = BRAT_model.unif_nystrom(Nystrom_subsample)
    BRAT_model.sketch_K()
    sigma_hat2 = BRAT_model.est_sigma_hat2(in_bag)
    sigma_hat = np.sqrt(sigma_hat2)
    lam = BRAT_model.learning_rate
    q = 1 - BRAT_model.dropout_rate
    s = (1 + lam * q) / lam
    for i, x in tqdm(test_points, desc="Evaluating coverage on test points", disable=disable_tqdm):
        y_pred = BRAT_model.predict(x.reshape(1, -1))[0]
        truth = ground_truth(x)
        rn_norm = BRAT_model.sketch_r(x)
        tau_hat = s * rn_norm * sigma_hat
        tau_hat2 = tau_hat**2
        ci = (y_pred - norm.ppf(1 - alpha / 2) * tau_hat, y_pred + norm.ppf(1 - alpha / 2) * tau_hat)
        
        covered = int(ci[0] <= truth <= ci[1])

        records.append({
            'test_point_idx': i,
            'y_pred': y_pred,
            "truth": truth,
            "ci_lower": ci[0],
            "ci_upper": ci[1],
            "width": ci[1] - ci[0],
            "sigma2_hat": sigma_hat2,
            "tau2_hat": tau_hat2,
            "covered": covered
        })

    df = pd.DataFrame(records)
    avg_coverage_rate = df["covered"].mean()

    return avg_coverage_rate, df

def PI_RI_coverage_rate(n_BRAT, total_trees, max_depth, learning_rate, 
                        subsample_rate, dropout_rate, min_sample_split, n_train, 
                        n_test, base_seed, noise_std, in_bag,
                        test_point, Nystrom_subsample=None, disable_tqdm=True, alpha=0.05):
    """
    Compute the coverage rates and bounds for Prediction Intervals (PI) and Reproduction Intervals (RI)
    at a fixed test point using multiple independently trained BRAT models.

    For each of n_BRAT replications:
        - A new synthetic training dataset is generated.
        - A BRAT model is trained on it.
        - The model's prediction at the fixed test point is recorded, along with its PI and RI.
        - Each model's PI and RI are evaluated by checking how many of the other models' predictions fall within its bounds.

    Returns:
        - Two separate DataFrames: one for PI-related statistics and one for RI-related statistics.
        - Each DataFrame includes the interval bounds, width, and cross-model coverage rate.

    Parameters:
        - n_BRAT (int): Number of independent BRAT replications to train.
        - total_trees (int): Number of trees in each BRAT model.
        - max_depth (int): Maximum depth of each decision tree.
        - subsample_rate (float): Fraction of samples used per tree.
        - dropout_rate (float): Dropout rate applied in the boosting process.
        - min_sample_split (int): Minimum number of samples required to split a node.
        - n_train (int): Number of training points per replication.
        - n_test (int): Number of test points (not used here but passed for consistency).
        - base_seed (int): Random seed for reproducibility.
        - noise_std (float): Standard deviation of the noise in the synthetic data.
        - in_bag (bool): Whether to use in-bag samples for estimating noise variance.
        - test_point (np.ndarray): The fixed input x at which PI and RI are computed.
        - disable_tqdm (bool): If True, disables tqdm progress bars.
        - alpha (float): Significance level for interval construction (default 0.05).

    Returns:
        - avg_pi_coverage (float): Mean prediction interval coverage rate over all models.
        - avg_ri_coverage (float): Mean reproduction interval coverage rate over all models.
        - BRAT_list (List): List of all trained BRAT models.
        - df_pi (pd.DataFrame): DataFrame containing:
            - 'idx': Model index
            - 'y_pred': Predicted value at the test point
            - 'pi_lower', 'pi_upper': PI bounds
            - 'sigma2_hat', 'tau2_hat': Variance estimates
            - 'pi_coverage': Proportion of other models' predictions inside this model's PI
            - 'pi_width': Width of the PI
        - df_ri (pd.DataFrame): DataFrame containing:
            - 'idx': Model index
            - 'y_pred': Predicted value at the test point
            - 'ri_lower', 'ri_upper': RI bounds
            - 'sigma2_hat', 'tau2_hat': Variance estimates
            - 'ri_coverage': Proportion of other models' predictions inside this model's RI
            - 'ri_width': Width of the RI
    """

    rng = np.random.RandomState(base_seed)
    seeds = rng.randint(low=0, high=2**32 - 1, size=n_BRAT)

    BRAT_list = []
    shared_rows = []
    
    for i in tqdm(range(n_BRAT), desc="Training BRAT models", disable=disable_tqdm):
        X_train, y_train, X_test, y_test = generate_data(
            function_type='friedman1', n_train=n_train, n_test=n_test,
            noise_std=noise_std, seed=seeds[i]
        )

        BRAT_model = BRATD(n_estimators=total_trees, learning_rate=learning_rate, max_depth=max_depth, 
                        min_samples_split=min_sample_split, subsample_rate=subsample_rate,
                        dropout_rate=dropout_rate, disable_tqdm=True)
        BRAT_model.fit(X_train, y_train, X_test, y_test)
        BRAT_list.append(BRAT_model)
        
        pi, _, ri, y_pred, _, sigma2_hat, tau2_hat = all_intervals(
            BRAT_model, X_train, y_train, X_test, y_test,
            test_point, in_bag=in_bag, alpha=alpha
        )

        shared_rows.append({
            "BRAT_idx": i,
            "y_pred": y_pred,
            "pi_lower": pi[0],
            "pi_upper": pi[1],
            "ri_lower": ri[0],
            "ri_upper": ri[1],
            "sigma2_hat": sigma2_hat,
            "tau2_hat": tau2_hat,
        })

    df = pd.DataFrame(shared_rows)

    # Evaluate PI & RI coverage separately
    pi_coverage = []
    ri_coverage = []

    for i in tqdm(range(n_BRAT), desc="Evaluating coverage rates", disable=disable_tqdm):
        pi_count = 0
        ri_count = 0
        for j in range(n_BRAT):
            if i == j:
                continue
            pred_j = df.loc[j, "y_pred"]
            if df.loc[i, "pi_lower"] <= pred_j <= df.loc[i, "pi_upper"]:
                pi_count += 1
            if df.loc[i, "ri_lower"] <= pred_j <= df.loc[i, "ri_upper"]:
                ri_count += 1
        denom = n_BRAT - 1
        pi_coverage.append(pi_count / denom)
        ri_coverage.append(ri_count / denom)

    # Construct separate DataFrames
    df_pi = df[["BRAT_idx", "y_pred", "pi_lower", "pi_upper", "sigma2_hat", "tau2_hat"]].copy()
    df_pi["pi_coverage"] = pi_coverage
    df_pi["pi_width"] = df_pi["pi_upper"] - df_pi["pi_lower"]

    df_ri = df[["BRAT_idx", "y_pred", "ri_lower", "ri_upper", "sigma2_hat", "tau2_hat"]].copy()
    df_ri["ri_coverage"] = ri_coverage
    df_ri["ri_width"] = df_ri["ri_upper"] - df_ri["ri_lower"]

    return np.mean(pi_coverage), np.mean(ri_coverage), BRAT_list, df_pi, df_ri