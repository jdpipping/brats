import numpy as np
import scipy.linalg as spl
from scipy import sparse

def compute_k_vector(BRAT_model, X_train, x):
    """
    Vectorized computation of the influence vector k at test point x using cached leaf assignments.

    Parameters:
      BRATD_model: Trained BRATD model.
      X_train: (n_samples, n_features)
      x: (n_features,)

    Returns:
      k_vector: (n_samples,) influence weights for x.
    """
    n, T = X_train.shape[0], len(BRAT_model.models)

    # Cached leaf assignments: shape (n_samples, T)
    leaf_ids_train = BRAT_model.leaf_assignments  # shape: (n, T)

    # Get leaf ID for test point x for all trees at once
    leaf_ids_x = np.array([tree.get_tree().apply(x.reshape(1, -1))[0] for tree in BRAT_model.models])  # shape: (T,)

    # Create same_leaf_mask: shape (n, T), True if train sample in same leaf as x
    same_leaf_mask = (leaf_ids_train == leaf_ids_x[None, :])  # shape: (n, T)

    # Subsample mask: shape (n, T)
    selected_mask = BRAT_model.subsample  # shape: (n, T)

    # Valid samples in same leaf AND selected in subsample
    valid_mask = same_leaf_mask & selected_mask  # shape: (n, T)

    # Count of valid samples per tree: shape: (T,)
    counts = np.sum(valid_mask, axis=0)  # shape: (T,)

    # Avoid division by zero, set counts to 1 where zero to avoid NaNs
    counts_safe = np.where(counts == 0, 1, counts)  # shape: (T,)

    # Normalize valid_mask by counts: shape: (n, T)
    valid_contributions = valid_mask / counts_safe  # shape: (n, T)

    # Zero out trees where count was 0 to avoid wrong sums
    valid_contributions[:, counts == 0] = 0.0

    # Average across trees
    k_vector = np.sum(valid_contributions, axis=1) / T  # shape: (n,)

    return k_vector


def compute_k_vector_batch(BRAT_model, X_train, X_batch):
    m = X_batch.shape[0]
    n = X_train.shape[0]
    C = np.zeros((m, n))
    
    for i in range(m):
        C[i, :] = compute_k_vector(BRAT_model, X_train, X_batch[i])
    return C


# find the K matrix for a given training set X_train and an ensemble of trees
def find_K_matrix(BRAT_model, X_train, Nystrom_subsample=None, reg=1e-6, rec=False):
    """
    Compute the influence matrix K or Nyström approximation for X_train.
    
    Parameters:
      BRATD_model: Trained BRATD model.
      X_train: Training data (n_samples, n_features).
      Nystrom_subsample: 
          - If None, returns full K. 
          - If float (0 < value <= 1), use Nyström with n_components = n * Nystrom_subsample.
      reg: Regularization for W matrix inversion.
      rec: If True, use recursive Nyström; else uniform Nyström.
    
    Returns:
      - Full K matrix (n x n), or
      - C (n x Nystrom_n), W (Nystrom_n x Nystrom_n), sampled_indices (Nyström).
    """
    import scipy.linalg as spl

    n = X_train.shape[0]
    B = len(BRAT_model.models)

    if Nystrom_subsample is None:
        # --- Full K computation ---
        ID = np.zeros((n, B), dtype=np.int32)
        for t, model in enumerate(BRAT_model.models):
            decision_tree = model.get_tree()
            ID[:, t] = decision_tree.apply(X_train)

        subsample_mask = BRAT_model.subsample.astype(float)

        eq = (ID[:, None, :] == ID[None, :, :]).astype(float)
        eq = eq * subsample_mask[None, :, :]
        row_sums = np.sum(eq, axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        eq_norm = eq / row_sums
        K = np.sum(eq_norm, axis=2) / B
        return K

    else:
        Nystrom_n = int(n * Nystrom_subsample)
        rng = np.random.default_rng()

        if not rec:
            # --- Uniform Nyström ---
            # Update the nystrom subsample indices to the model instantiation.
            sampled_indices = rng.choice(n, size=Nystrom_n, replace=False)
            BRAT_model.nys_sub = sampled_indices
            C = compute_k_vector_batch(BRAT_model, X_train, X_train[sampled_indices])  # shape: (Nystrom_n, n)
            C = C.T  # shape: (n, Nystrom_n)

            W = C[sampled_indices, :]  # shape: (Nystrom_n, Nystrom_n)

            return C, W, sampled_indices

        else:
            # --- Recursive Nyström ---
            n_oversample = np.log(Nystrom_n)
            k = int(np.ceil(Nystrom_n / (4 * n_oversample)))
            n_levels = int(np.ceil(np.log(n / Nystrom_n) / np.log(2)))
            perm = rng.permutation(n)

            size_list = [n]
            for l in range(1, n_levels + 1):
                size_list.append(int(np.ceil(size_list[l - 1] / 2)))

            sample = np.arange(size_list[-1])
            indices = perm[sample]
            weights = np.ones(indices.shape[0])

            # Precompute diagonal of kernel matrix
            k_diag = np.array([compute_k_vector(BRAT_model, X_train, X_train[i])[i] for i in range(n)])

            for l in reversed(range(n_levels)):
                current_indices = perm[:size_list[l]]

                # Compute KS and SKS using compute_k_vector
                KS = np.zeros((len(current_indices), len(indices)))
                for i, idx in enumerate(indices):
                    KS[:, i] = compute_k_vector(BRAT_model, X_train, X_train[idx])[current_indices]

                SKS = KS[sample, :]

                if k >= SKS.shape[0]:
                    lmbda = 10e-6
                else:
                    weighted_SKS = SKS * weights[:, None] * weights[None, :]
                    eigs = spl.eigvalsh(weighted_SKS, subset_by_index=(SKS.shape[0]-k, SKS.shape[0]-1))
                    lmbda = (np.sum(np.diag(SKS) * (weights ** 2)) - np.sum(eigs)) / k

                lmbda = max(lmbda, 1e-6 * SKS.shape[0])

                R = np.linalg.solve(SKS + np.diag(lmbda * weights ** (-2)), KS.T).T

                if l != 0:
                    leverage_score = np.minimum(1.0, n_oversample * (1 / lmbda) * np.maximum(0.0, (
                            k_diag[current_indices] - np.sum(R * KS, axis=1))))
                    sample = np.where(rng.uniform(size=size_list[l]) < leverage_score)[0]
                    if sample.size == 0:
                        leverage_score[:] = Nystrom_n / size_list[l]
                        sample = rng.choice(size_list[l], size=Nystrom_n, replace=False)
                    weights = np.sqrt(1. / leverage_score[sample])
                else:
                    leverage_score = np.minimum(1.0, (1 / lmbda) * np.maximum(0.0, (
                            k_diag[current_indices] - np.sum(R * KS, axis=1))))
                    p = leverage_score / leverage_score.sum()
                    sample = rng.choice(n, size=Nystrom_n, replace=False, p=p)

                indices = perm[sample]

            # Final C and W for Recursive Nyström
            C = np.zeros((n, Nystrom_n))
            for i, idx in enumerate(indices):
                C[:, i] = compute_k_vector(BRAT_model, X_train, X_train[idx])

            W = np.zeros((Nystrom_n, Nystrom_n))
            for i, idx_i in enumerate(indices):
                W[i, :] = C[idx_i, :]

            BRAT_model.nys_sub = indices
            return C, W, indices

def calculate_rn(BRAT, X_train, x, K=None, sketched_inverse_K_sq=None):
    """
    Compute r_n and its norm.
    
    If K is provided, use full matrix inverse.
    If C and W are provided, use Woodbury identity (Nyström approximation).
    
    Parameters:
      BRATD: Trained BRATD model.
      k_vector: Influence vector (n,).
      K: Full influence matrix (n, n) [optional].
      C: Nyström matrix (n, m) [optional].
      W: Nyström W matrix (m, m) [optional].
    
    Returns:
      r_n: Estimated r_n vector (n,).
      r_norm: L2 norm of r_n.
    """
    lam = BRAT.learning_rate
    q = 1 - BRAT.dropout_rate

    if K is not None:
        # Full K: use direct inverse
        KRR = np.linalg.inv(q * K + (1 / lam) * np.eye(K.shape[0]))
        k_vector = compute_k_vector(BRAT, X_train, x)
        rn_norm = np.linalg.norm(KRR @ k_vector)
    elif sketched_inverse_K_sq is not None:
        # Nyström + Woodbury
        sketched_k_vector = compute_k_vector_batch(BRAT, X_train[BRAT.nys_sub], x)
        rn_norm = np.sqrt(sketched_k_vector.T @ sketched_inverse_K_sq @ sketched_k_vector)
    else:
        raise ValueError("Either K or sketched K must be provided.")

    return rn_norm



def estimate_noise_variance(BRAT_model, X_train, y_train, X_test, y_test, in_bag = False):
    """
    Estimate the noise variance sigma_epsilon^2 from training residuals.
    
    Parameters:
      BRATD_model: A trained BRATD model.
      X_train: Training features as a numpy array.
      y_train: Training responses as a numpy array.
    
    Returns:
      sigma2_hat: An estimate of the noise variance.
    """
    if in_bag:
        y_pred_train = BRAT_model.predict(X_train)
        residuals = y_train - y_pred_train
        sigma2_hat = np.var(residuals, ddof=1)
    else:
        y_pred_test = BRAT_model.predict(X_test)
        residuals = y_test - y_pred_test
        sigma2_hat = np.var(residuals, ddof=1)

    return sigma2_hat

def estimate_built_in_variance(BRAT_model, X_train, y_train, X_test, y_test, x, in_bag=False, Nystrom_subsample=None, rec=False):
    """
    Estimate the noise variance using the built-in method of the BRATD model.
    
    Parameters:
      BRATD_model: A trained BRATD model.
      X_train: Training features.
      y_train: Training responses.
      X_test: Test features.
      y_test: Test responses.
      x: A test point.
      in_bag: Use in-bag samples for noise variance.
      Nystrom_subsample: If None, use full K. If float (0 < value <= 1), use Nyström with n_components = n * Nystrom_subsample.
    
    Returns:
      If Nystrom:
        C: The subsample columns.
        W: The subsample kernels.
        rn: The response weighting vector.
        rn_norm: The norm of the response weighting vector.
        sigma2_hat: Noise variance estimate.
        tau2_hat: Built-in estimates of the ensemble's variance.
      If full kernel:
        K: The full kernel.
        rn: The response weighting vector.
        rn_norm: The norm of the response weighting vector.
        sigma2_hat: Noise variance estimates..
        tau2_hat: Built-in estimates of the ensemble's variance.
    """

    sigma2_hat = estimate_noise_variance(BRAT_model, X_train, y_train, X_test, y_test, in_bag=in_bag)
    k_vector = compute_k_vector(BRAT_model, X_train, x)\
    
    q = 1 - BRAT_model.dropout_rate
    lam = BRAT_model.learning_rate

    if Nystrom_subsample is not None:
        C, W, _ = find_K_matrix(BRAT_model, X_train, Nystrom_subsample=Nystrom_subsample, rec=rec)
        rn, rn_norm = calculate_rn(BRAT_model, k_vector, C=C, W=W)
        tau2_hat = (1 + lam * q) ** 2 / lam ** 2 * sigma2_hat ** 2 * rn_norm ** 2
        return C, W, rn, rn_norm, sigma2_hat, tau2_hat
    else:
        K = find_K_matrix(BRAT_model, X_train)
        rn, rn_norm = calculate_rn(BRAT_model, k_vector, K=K)
        tau2_hat = (1 + lam * q) ** 2 / lam ** 2 * sigma2_hat ** 2 * rn_norm ** 2
        return  K, rn, rn_norm, sigma2_hat, tau2_hat

def estimate_emp_rep_variance(BRATD_model, x):
    """
    Estimate the between-replication variance (tau^2) for a given test point x.
    
    This function computes the variance of the predictions from each base learner in the BRATD ensemble.
    The idea is that the variation among individual ensemble predictions provides an estimate
    of the heterogeneity in replicated estimates.
    
    Parameters:
      BRATD_model: A trained BRATD model that contains an attribute which is an iterable
                 of base learners.
      x: A test point as a numpy array (shape: (n_features,)).
      
    Returns:
      tau2_hat: The estimated between-replication variance (a non-negative float).
    """
    
    predictions = []
    for est in BRATD_model.models:
        pred = est.predict(np.expand_dims(x, axis=0))[0]
        predictions.append(pred)
        
    predictions = np.array(predictions)
    
    tau2_hat = np.var(predictions, ddof=1)  
    
    return tau2_hat


