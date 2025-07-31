import sys
import numpy as np
import scipy.linalg as spl
import warnings
from BRAT.trees import SubsampledDecisionTreeRegressor
from tqdm import tqdm

from BRAT.variance_estimation import compute_k_vector, compute_k_vector_batch

class BRATD:
    """
    Boulevard Regularized Additive Regression Trees with Dropout (BRAT-D).

    This class implements an additive ensemble of decision trees where, at
    each boosting iteration, a random subset of previous trees is “dropped”
    (i.e. excluded) when computing the residuals.
    """
    def __init__(self, n_estimators=10, 
                 learning_rate=1.0, 
                 max_depth=4, 
                 min_samples_split=2, 
                 subsample_rate=0.8, 
                 dropout_rate=0.5, 
                 disable_tqdm=False):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.subsample_rate = subsample_rate
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.K =None
        self.C = None
        self.W = None
        self.coef = None
        self.nys_sub = None
        self.sketched_inverse_K_sq = None
        self.sigma_hat2 = None
        """
        Initialize the BRAT-D ensemble.

        Parameters
        ----------
        n_estimators
            Total number of trees to fit.
        learning_rate
            Shrinkage factor applied to each new tree’s predictions.
        max_depth
            Maximum depth for each decision tree.
        min_samples_split
            Minimum number of samples required to split an internal node.
        subsample_rate
            Fraction of training samples to draw (without replacement) for each tree.
        dropout_rate
            Fraction of already-fitted trees to “drop” when forming residuals.
            If 1.0, this reduces to a demeaned random forest (no boosting).
        disable_tqdm
            If True, do not show the tqdm progress bar during fitting.

        Raises
        ------
        Warning
            If `dropout_rate == 1.0` but `learning_rate < 1.0`, which
            is a demeaned random forests, we reset learning_rate to 1.0 and issue a Warning.
        """
        if self.dropout_rate == 1.0:
            if self.learning_rate < 1.0:
                self.learning_rate = 1.0
                raise Warning("Can't set learning rate less than 1 when dropout rate is 1. That's a demeaned random forest.")
        else:
            self.learning_rate = learning_rate

        self.models = []
        self.disable_tqdm = disable_tqdm
        self.yhat = None

    def fit(self, X_train, y_train, X_test, y_test):
        """
        Fit the BRAT-D model to training data, tracking test MSE at each step.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Training targets.
        X_test : array-like of shape (m_samples, n_features)
            Held-out features for monitoring mse.
        y_test : array-like of shape (m_samples,)
            Held-out targets for monitoring mse.

        Returns
        -------
        mse_list : list of float
            Test-set mean squared error after each tree is added.
        """
        n_samples = X_train.shape[0]
        total_trees = self.n_estimators
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.models = []
        self.subsample = np.zeros((len(y_train), self.n_estimators), dtype = bool)
        self.leaf_assignments = np.zeros((n_samples, self.n_estimators), dtype=int)
        mse_list = []
        
        pbar = tqdm(range(1, total_trees + 1), 
                    desc="Building BRATD trees", 
                    disable=self.disable_tqdm, 
                    file=sys.stdout)
        for b in pbar:
            if len(self.models) > 0:
                num_residual_models = int(np.round((1-self.dropout_rate) * len(self.models)))
                residual_used_tree_indices = np.random.permutation(len(self.models))[:num_residual_models]
                residual_models = [self.models[i] for i in residual_used_tree_indices]
            else:
                residual_models = []

            if residual_models:
                preds = np.zeros(n_samples, dtype=float)
                for model in residual_models:
                    preds += model.predict(X_train)
                preds /= len(self.models)
                preds *= self.learning_rate
                residuals = y_train - preds
            else:
                residuals = y_train

            tree = SubsampledDecisionTreeRegressor(subsample_rate=self.subsample_rate, max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X_train, y=residuals)
            self.models.append(tree)
            self.subsample[:, b-1] = tree.subsample
            self.leaf_assignments[:,b-1] = tree.leaf_assignments

            y_pred = self.predict(X_test)
            mse = np.mean((y_test - y_pred) ** 2).item()
            mse_list.append(mse)

            pbar.refresh()
            sys.stdout.flush()
        return mse_list

    def predict(self, x):
        """
        Predict regression targets for new data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features to predict.

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Predicted targets.
        """
        preds = np.zeros(x.shape[0], dtype=float)
        for tree in self.models:
            preds += self.learning_rate * tree.predict(x)
        if len(self.models) > 0:
            preds /= len(self.models)
        else:
            preds = np.zeros(x.shape[0], dtype=float)
        preds = preds * (1+ self.learning_rate * (1-self.dropout_rate)) / self.learning_rate
        return preds
    
    def full_K(self):
        """
        Compute the full expected tree-kernel matrix K for the training set.

        For each pair of training samples (i, j),  
        K[i, j] = average over trees t of the vote of sample j on sample i. 
        Hence K is symmetric and positive semi-definite.

        Returns
        -------
        K : np.ndarray of shape (n_samples, n_samples)
            The full influence/kernel matrix.
        """
        n = self.X_train.shape[0]
        B = len(self.models)
        ID = np.zeros((n, B), dtype=np.int32)
        for t, model in enumerate(self.models):
            decision_tree = model.get_tree()
            ID[:, t] = decision_tree.apply(self.X_train)

        subsample_mask = self.subsample.astype(float)

        eq = (ID[:, None, :] == ID[None, :, :]).astype(float)
        eq = eq * subsample_mask[None, :, :]
        row_sums = np.sum(eq, axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        eq_norm = eq / row_sums
        K = np.sum(eq_norm, axis=2) / B
        self.K = K
        return K
    
    def unif_nystrom(self, Nystrom_subsample):
        """
        Uniform Nyström approximation of the full kernel matrix.

        Randomly samples a subset of training points to form low-rank factors C and W.

        Parameters
        ----------
        Nystrom_subsample : float
            Fraction of training points to sample (0 < Nystrom_subsample ≤ 1).

        Returns
        -------
        C : np.ndarray, shape (n_samples, m)
            Cross-kernel between all training points and the m sampled landmark points.
        W : np.ndarray, shape (m, m)
            Kernel among the m sampled landmark points.
        sampled_indices : np.ndarray, shape (m,)
            Indices of the landmark points in the original training set.
        """
        X_train = self.X_train
        n = X_train.shape[0]

        Nystrom_n = int(n * Nystrom_subsample)
        rng = np.random.default_rng()

        max_retries = 5  # Limit the number of retries
        for retry in range(max_retries):
            # Sample indices without replacement
            sampled_indices = rng.choice(n, size=Nystrom_n, replace=False)
            self.nys_sub = sampled_indices

            # Compute C and W
            C = compute_k_vector_batch(self, X_train, X_train[sampled_indices])  # shape: (Nystrom_n, n)
            C = C.T  # shape: (n, Nystrom_n)
            W = C[sampled_indices, :]  # shape: (Nystrom_n, Nystrom_n)

            self.C = C
            self.W = W
            return C, W, sampled_indices

    def rec_nystrom(self, X_train, Nystrom_subsample=None, reg=1e-6):
        """
        Compute the influence matrix K or Nyström approximation for X_train.

        Parameters
        ----------
          X_train: Training data (n_samples, n_features).
          Nystrom_subsample: 
              - If None, returns full K. 
              - If float (0 < value <= 1), use Nyström with n_components = n * Nystrom_subsample.
          reg: Regularization for W matrix inversion.

        Returns:
        -------
          - C (n x Nystrom_n), W (Nystrom_n x Nystrom_n), sampled_indices (Nyström).
        """
        n = X_train.shape[0]
        B = len(self.models)

        Nystrom_n = int(n * Nystrom_subsample)
        rng = np.random.default_rng()

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
        k_diag = np.array([compute_k_vector(self, X_train, X_train[i])[i] for i in range(n)])

        for l in reversed(range(n_levels)):
            current_indices = perm[:size_list[l]]

            # Compute KS and SKS using compute_k_vector
            KS = np.zeros((len(current_indices), len(indices)))
            for i, idx in enumerate(indices):
                KS[:, i] = compute_k_vector(self, X_train, X_train[idx])[current_indices]

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
            C[:, i] = compute_k_vector(self, X_train, X_train[idx])

        W = np.zeros((Nystrom_n, Nystrom_n))
        for i, idx_i in enumerate(indices):
            W[i, :] = C[idx_i, :]

        self.C = C
        self.W = W
        self.nys_sub = indices
        return C, W, indices

    def sketch_k(self, x):
        """
        Compute the Nystrom-sketched kernel vector between a new point x
        and the training set, using the sampled subset self.nys_sub.

        Steps:
        1. Let s = number of Nyström samples (len(self.nys_sub)).
        2. Let T = total number of trees in the fitted model.
        3. Extract leaf assignments for the Nyström samples: 
            leaf_ids_train is an (s by T) array of leaf indices.
        4. For the test point x, get its leaf index in each tree:
            leaf_ids_x is a length-T vector.
        5. Build a boolean mask same_leaf_mask (s by T) that is True
            when a Nyström sample and x fall in the same leaf of tree t.
        6. Extract the original subsampling mask for those s samples,
            selected_mask (s by T).
        7. valid_mask = same_leaf_mask AND selected_mask; only count
            in-bag samples in matching leaves.
        8. For each tree t, count how many valid samples there are:
            counts[t] = sum over the n rows of valid_mask[:, t].
        9. To prevent division by zero, set any zero count to 1.
        10. Divide each column of valid_mask by its count, giving
            valid_contributions (n by T).
        11. Zero out all contributions for trees where the count was zero.
        12. Finally average across the T columns to get a length-n vector:
            sketched_k[i] = (1/T) * sum_{t=1..T} valid_contributions[i, t].

        Returns:
        sketched_k: a numpy array of shape (n,), the Nyström-sketched
                    kernel evaluations between x and each sampled train point.
        """
        n, T = self.X_train[self.nys_sub].shape[0], len(self.models)

        # Cached leaf assignments: shape (n_samples, T)
        leaf_ids_train = self.leaf_assignments[self.nys_sub]  # shape: (Nystrom_n, T)

        # Get leaf ID for test point x for all trees at once
        leaf_ids_x = np.array([tree.get_tree().apply(x.reshape(1, -1))[0] for tree in self.models])  # shape: (T,)

        # Create same_leaf_mask: shape (Nystrom_n, T), True if train sample in same leaf as x
        same_leaf_mask = (leaf_ids_train == leaf_ids_x[None, :])  # shape: (Nystrom_n, T)

        # Subsample mask: shape (n, T)
        selected_mask = self.subsample[self.nys_sub]  # shape: (Nystrom_n, T)

        # Valid samples in same leaf AND selected in subsample
        valid_mask = same_leaf_mask & selected_mask  # shape: (Nystrom_n, T)

        # Count of valid samples per tree: shape: (T,)
        counts = np.sum(valid_mask, axis=0)  # shape: (T,)

        # Avoid division by zero, set counts to 1 where zero to avoid NaNs
        counts_safe = np.where(counts == 0, 1, counts)  # shape: (T,)

        # Normalize valid_mask by counts: shape: (n, T)
        valid_contributions = valid_mask / counts_safe  # shape: (Nystrom_n, T)

        # Zero out trees where count was 0 to avoid wrong sums
        valid_contributions[:, counts == 0] = 0.0

        # Average across trees
        sketched_k = np.sum(valid_contributions, axis=1) / T  # shape: (Nystrom_n,)

        return sketched_k

    def sketch_K(self):
        """
        Compute a sketched approximation to the inverse tree-kernel matrix K
        (and store its square), using the Woodbury identity on the Nyström factors.

        Requires that self.W (s*s) and self.C (n*s) are already set
        via a prior call to unif_nystrom or rec_nystrom. Make sure you ran:
        self.unif_nystrom, self.rec_nystrom or self.full_K before calling this function
        The result is stored in self.sketched_inverse_K_sq (s*s).

        Avoid training small ensemble on large dataset. This will likely result in similar voting vectors and
        increase the chance for the landmark matrix W to be singular. 
        """
        W = self.W
        C = self.C
        lam = self.learning_rate
        q = 1 - self.dropout_rate
        retry_count = 0

        while retry_count < 10:
            try:
                W_inv = np.linalg.pinv(W)
                break
            except np.linalg.LinAlgError:
                retry_count += 1
                if retry_count == 10:
                    raise RuntimeError("SVD did not converge after 10 attempts.")
        CTC = C.T @ C
        try:
            sketched_inverse_K = (W_inv @ C.T @ (lam * np.eye(C.shape[0])) 
                                - lam**2 * (W_inv @ CTC @ np.linalg.inv((1/q) * W + lam * CTC)) @ C.T)
            sketched_inverse_K_sq = sketched_inverse_K @ sketched_inverse_K.T
            self.sketched_inverse_K_sq = sketched_inverse_K_sq

        except np.linalg.LinAlgError:
            print("Singular matrix encountered during sketching. Falling back to full K.")
            K = self.full_K()
        
    def sketch_r(self, x, vector=False):
        """
        Compute the BRAT weight for a new point x,
        either from the sketched inverse (if available) or exactly.

        Parameters
        ----------
        x : array-like, shape (n_features,)
            A single test point.

        vector : bool, default=False
            If True, also return the raw influence vector (sketched_k or rn)
            along with its norm.

        Returns
        -------
        rn_norm : float
            The clipped norm of the (approximate) influence vector, capped at 10.

        If vector=True, returns a tuple (r_vec, rn_norm), where
        r_vec is the m-dim landmark-space vector (if sketched) or
            the full n-vector rn (if using the exact K).
        """
        if self.sketched_inverse_K_sq is not None and self.K is None:  
            sketched_k = self.sketch_k(x)
            rn_norm = np.sqrt(sketched_k.T @ self.sketched_inverse_K_sq @ sketched_k)
            if rn_norm > 10:
                rn_norm = np.array([1.0])
            return rn_norm
        elif self.sketched_inverse_K_sq is None and self.K is not None:
            k = compute_k_vector(self, self.X_train, x)
            rn = np.linalg.solve((1/self.learning_rate) * np.eye(self.K.shape[0]) + (1-self.dropout_rate) * self.K, k)
            rn_norm = np.linalg.norm(rn)
            if rn_norm > 10:
                rn_norm = np.array([1.0])
            if vector:
                return rn, rn_norm
            else:
                return rn_norm
    
    def est_sigma_hat2(self, in_bag):
        """
        Estimate the variance of the noise.
        Parameters
        ----------
        in_bag: Boolean. If True, will use the training set to estimate the variance

        Return
        ------
        sigma_hat2: Estimated variance.
        """
        X_train = self.X_train
        y_train = self.y_train
        X_test = self.X_test
        y_test = self.y_test
        if in_bag:
            n = X_train.shape[0]
            mse = np.mean((y_train - self.predict(X_train)) ** 2)
            sigma_hat2 = mse * (n/ (n - 1))
        else:
            n = X_test.shape[0]
            mse = np.mean((y_test - self.predict(X_test)) ** 2)
            sigma_hat2 = mse * (n/ (n - 1))
        self.sigma_hat2 = sigma_hat2
        return sigma_hat2

    def est_tau_hat2(self, in_bag, Nystrom_subsample, x):
        """
        Estimate the built-in variance tau2.

        Parameters
        ----------
        in_bag: Boolean. Deciding the estimation of sigma2_hat
        Nystrom_subsample: Nystrom subsample rate used to estimate the BRAT weight vector.
        x: A point of interest of making inference.

        Return
        ------
        sigma_hat2: Cache estiamted noise variane
        rn_norm: Cache BRAT weight vector norm
        tau_hat2: Estimated built-in variance.
        """
        sigma_hat2 = self.est_sigma_hat2(in_bag)
        self.unif_nystrom(Nystrom_subsample)
        self.sketch_K()
        rn_norm = self.sketch_r(x)
        s = (1+self.learning_rate * (1-self.dropout_rate)) / self.learning_rate
        tau_hat2 = (s * rn_norm * sigma_hat2)
        return sigma_hat2, rn_norm, tau_hat2
    
import sys
import numpy as np
import warnings
from tqdm import tqdm

from BRAT.trees import SubsampledDecisionTreeRegressor
from BRAT.variance_estimation import compute_k_vector, compute_k_vector_batch


class BRATP:
    """
    Boulevard Regularized Additive Regression Trees with Parallelized Training (BRAT-P).

    This class implements an additive ensemble of decision trees where, at
    each boosting iteration, you drop one “group” of trees when forming
    residuals (the “permutation” variant of BRAT).
    """

    def __init__(
        self,
        n_estimators=10,
        learning_rate=1.0,
        max_depth=4,
        min_samples_split=2,
        subsample_rate=0.8,
        n_trees_per_group=10,
        disable_tqdm=False,
        drop_first_row=False,
    ):
        """
        Initialize the BRAT-P model.

        Parameters
        ----------
        n_estimators : int
            Total number of trees to fit.
        learning_rate : float
            Scale factor applied to each new trees' predictions.
        max_depth : int
            Maximum depth for each DecisionTreeRegressor.
        min_samples_split : int
            Minimum samples to split a node.
        subsample_rate : float
            Fraction of training samples to draw per tree.
        n_trees_per_group : int
            Number of trees in each boosting round.
        disable_tqdm : bool
            If True, turns off the progress bar when training individual BRATP model.
        drop_first_row : bool
            If True, we build a demeaned random forest in the first row instead of a gradient boosting tree ensemble.
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.subsample_rate = subsample_rate
        self.n_trees_per_group = n_trees_per_group
        self.disable_tqdm = disable_tqdm
        self.drop_first_row = drop_first_row

        # number of full groups
        self.num_groups = (n_estimators + n_trees_per_group - 1) // n_trees_per_group

        # placeholders
        self.models = []              # flat list of trees
        self.trees_table = None       # list-of-lists of trees
        self.tree_pred_table = None   # 3D array of in-bag train preds
        self.subsample = None         # in-bag mask, shape=(n_train, n_estimators)

        # for inference
        self.K = None                 # full kernel matrix
        self.C = None                 # nystrom cross-kernel
        self.W = None                 # nystrom landmark kernel
        self.nys_sub = None           # landmark indices
        self.sketched_inverse_K_sq = None

        # record test errors
        self.mse_values = []

    def fit(self, X_train, y_train, X_test, y_test):
        """
        Fit the BRAT-P model, recording test MSE at each tree.

        Parameters
        ----------
        X_train : array-like, shape=(n_train, n_features)
            Training features.
        y_train : array-like, shape=(n_train,)
            Training targets.
        X_test : array-like, shape=(n_test, n_features)
            Held-out features for MSE monitoring.
        y_test : array-like, shape=(n_test,)
            Held-out targets.

        Returns
        -------
        mse_values : list of float
            Test MSE after each tree is added.
        """
        n_train = X_train.shape[0]
        B = self.n_estimators
        tpq = self.n_trees_per_group
        ng = self.num_groups

        # initialize storage
        self.models = []
        self.subsample = np.zeros((n_train, B), dtype=bool)
        self.trees_table = [
            [
                SubsampledDecisionTreeRegressor(
                    subsample_rate=self.subsample_rate,
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split
                )
                for _ in range(tpq)
            ]
            for _ in range(ng)
        ]
        self.tree_pred_table = np.zeros((ng, tpq, n_train), dtype=float)
        self.mse_values = []

        pbar = tqdm(
            range(1, B + 1),
            desc="Building BRAT-P trees",
            disable=self.disable_tqdm,
            file=sys.stdout
        )

        for b in pbar:
            group = (b - 1) // tpq
            slot = (b - 1) % tpq

            # form residuals by dropping this slot in all groups
            R = self.tree_pred_table.copy()
            R[:, slot, :] = 0.0
            if group == 0 and self.drop_first_row:
                R[0, :, :] = 0.0
            elif group > 0:
                R[group, :, :] = 0.0

            # compute dropped prediction
            mask = (R != 0).any(axis=2)
            avg_partial = np.zeros((tpq, n_train), dtype=float)
            for t in range(tpq):
                valid_rows = mask[:, t]
                if valid_rows.sum() > 0:
                    avg_partial[t] = R[valid_rows, t, :].mean(axis=0)
            dropped = avg_partial.sum(axis=0)
            residual = y_train - dropped

            # fit new tree
            tree = SubsampledDecisionTreeRegressor(
                subsample_rate=self.subsample_rate,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split
            )
            tree.fit(X_train, residual)
            self.trees_table[group][slot] = tree
            self.models.append(tree)

            # store its train prediction
            pred_train = tree.predict(X_train)
            if b == 1:
                self.tree_pred_table[group][slot] = self.learning_rate * pred_train
            else:
                self.tree_pred_table[group][slot] = pred_train

            # record test MSE
            y_pred_test = self.predict(X_test, num_trees=b)
            mse = np.mean((y_test - y_pred_test) ** 2)
            self.mse_values.append(mse)

            pbar.refresh()

        return self.mse_values

    def predict(self, x, num_trees=None):
        """
        Predict regression targets with the first `num_trees` trees.

        Parameters
        ----------
        X : array-like, shape=(n_samples, n_features)
        num_trees : int or None
            If None, use all trees; otherwise use first `num_trees`.

        Returns
        -------
        y_pred : np.ndarray, shape=(n_samples,)
        """
        B = num_trees or self.n_estimators
        tpq = self.n_trees_per_group
        ng = (B + tpq - 1) // tpq
        n_samples = x.shape[0]

        P = np.zeros((ng, tpq, n_samples), dtype=float)
        for g in range(ng):
            for t in range(tpq):
                idx = g * tpq + t
                if idx < B:
                    P[g, t] = self.trees_table[g][t].predict(x)

        avg_over_groups = np.nanmean(P, axis=0)
        return np.nansum(avg_over_groups, axis=0)

    def full_K(self):
        """
        Compute the full expected tree-kernel matrix K for the training set.

        For each pair of training samples (i, j),  
        K[i, j] = average over trees t of the vote of sample j on sample i. 
        Hence K is symmetric and positive semi-definite.

        Returns
        -------
        K : np.ndarray of shape (n_samples, n_samples)
            The full influence/kernel matrix.
        """
        n = self.X_train.shape[0]
        B = len(self.models)
        ID = np.zeros((n, B), dtype=np.int32)
        for t, model in enumerate(self.models):
            decision_tree = model.get_tree()
            ID[:, t] = decision_tree.apply(self.X_train)

        subsample_mask = self.subsample.astype(float)

        eq = (ID[:, None, :] == ID[None, :, :]).astype(float)
        eq = eq * subsample_mask[None, :, :]
        row_sums = np.sum(eq, axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        eq_norm = eq / row_sums
        K = np.sum(eq_norm, axis=2) / B
        self.K = K
        return K

    def unif_nystrom(self, Nystrom_subsample):
        """
        Uniform Nyström approximation of the full kernel matrix.

        Randomly samples a subset of training points to form low-rank factors C and W.

        Parameters
        ----------
        Nystrom_subsample : float
            Fraction of training points to sample (0 < Nystrom_subsample ≤ 1).

        Returns
        -------
        C : np.ndarray, shape (n_samples, m)
            Cross-kernel between all training points and the m sampled landmark points.
        W : np.ndarray, shape (m, m)
            Kernel among the m sampled landmark points.
        sampled_indices : np.ndarray, shape (m,)
            Indices of the landmark points in the original training set.
        """
        X_train = self.X_train
        n = X_train.shape[0]

        Nystrom_n = int(n * Nystrom_subsample)
        rng = np.random.default_rng()

        max_retries = 5  # Limit the number of retries
        for retry in range(max_retries):
            # Sample indices without replacement
            sampled_indices = rng.choice(n, size=Nystrom_n, replace=False)
            self.nys_sub = sampled_indices

            # Compute C and W
            C = compute_k_vector_batch(self, X_train, X_train[sampled_indices])  # shape: (Nystrom_n, n)
            C = C.T  # shape: (n, Nystrom_n)
            W = C[sampled_indices, :]  # shape: (Nystrom_n, Nystrom_n)

            self.C = C
            self.W = W
            return C, W, sampled_indices

    def rec_nystrom(self, X_train, Nystrom_subsample=None, reg=1e-6):
        """
        Compute the influence matrix K or Nyström approximation for X_train.

        Parameters
        ----------
          X_train: Training data (n_samples, n_features).
          Nystrom_subsample: 
              - If None, returns full K. 
              - If float (0 < value <= 1), use Nyström with n_components = n * Nystrom_subsample.
          reg: Regularization for W matrix inversion.

        Returns:
        -------
          - C (n x Nystrom_n), W (Nystrom_n x Nystrom_n), sampled_indices (Nyström).
        """
        n = X_train.shape[0]
        B = len(self.models)

        Nystrom_n = int(n * Nystrom_subsample)
        rng = np.random.default_rng()

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
        k_diag = np.array([compute_k_vector(self, X_train, X_train[i])[i] for i in range(n)])

        for l in reversed(range(n_levels)):
            current_indices = perm[:size_list[l]]

            # Compute KS and SKS using compute_k_vector
            KS = np.zeros((len(current_indices), len(indices)))
            for i, idx in enumerate(indices):
                KS[:, i] = compute_k_vector(self, X_train, X_train[idx])[current_indices]

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
            C[:, i] = compute_k_vector(self, X_train, X_train[idx])

        W = np.zeros((Nystrom_n, Nystrom_n))
        for i, idx_i in enumerate(indices):
            W[i, :] = C[idx_i, :]

        self.C = C
        self.W = W
        self.nys_sub = indices
        return C, W, indices

    def sketch_k(self, x):
        """
        Compute the Nystrom-sketched kernel vector between a new point x
        and the training set, using the sampled subset self.nys_sub.

        Steps:
        1. Let s = number of Nyström samples (len(self.nys_sub)).
        2. Let T = total number of trees in the fitted model.
        3. Extract leaf assignments for the Nyström samples: 
            leaf_ids_train is an (s by T) array of leaf indices.
        4. For the test point x, get its leaf index in each tree:
            leaf_ids_x is a length-T vector.
        5. Build a boolean mask same_leaf_mask (s by T) that is True
            when a Nyström sample and x fall in the same leaf of tree t.
        6. Extract the original subsampling mask for those s samples,
            selected_mask (s by T).
        7. valid_mask = same_leaf_mask AND selected_mask; only count
            in-bag samples in matching leaves.
        8. For each tree t, count how many valid samples there are:
            counts[t] = sum over the n rows of valid_mask[:, t].
        9. To prevent division by zero, set any zero count to 1.
        10. Divide each column of valid_mask by its count, giving
            valid_contributions (n by T).
        11. Zero out all contributions for trees where the count was zero.
        12. Finally average across the T columns to get a length-n vector:
            sketched_k[i] = (1/T) * sum_{t=1..T} valid_contributions[i, t].

        Returns:
        sketched_k: a numpy array of shape (n,), the Nyström-sketched
                    kernel evaluations between x and each sampled train point.
        """
        n, T = self.X_train[self.nys_sub].shape[0], len(self.models)

        # Cached leaf assignments: shape (n_samples, T)
        leaf_ids_train = self.leaf_assignments[self.nys_sub]  # shape: (Nystrom_n, T)

        # Get leaf ID for test point x for all trees at once
        leaf_ids_x = np.array([tree.get_tree().apply(x.reshape(1, -1))[0] for tree in self.models])  # shape: (T,)

        # Create same_leaf_mask: shape (Nystrom_n, T), True if train sample in same leaf as x
        same_leaf_mask = (leaf_ids_train == leaf_ids_x[None, :])  # shape: (Nystrom_n, T)

        # Subsample mask: shape (n, T)
        selected_mask = self.subsample[self.nys_sub]  # shape: (Nystrom_n, T)

        # Valid samples in same leaf AND selected in subsample
        valid_mask = same_leaf_mask & selected_mask  # shape: (Nystrom_n, T)

        # Count of valid samples per tree: shape: (T,)
        counts = np.sum(valid_mask, axis=0)  # shape: (T,)

        # Avoid division by zero, set counts to 1 where zero to avoid NaNs
        counts_safe = np.where(counts == 0, 1, counts)  # shape: (T,)

        # Normalize valid_mask by counts: shape: (n, T)
        valid_contributions = valid_mask / counts_safe  # shape: (Nystrom_n, T)

        # Zero out trees where count was 0 to avoid wrong sums
        valid_contributions[:, counts == 0] = 0.0

        # Average across trees
        sketched_k = np.sum(valid_contributions, axis=1) / T  # shape: (Nystrom_n,)

        return sketched_k


    def sketch_K(self):
        """
        Compute a sketched approximation to the inverse tree-kernel matrix K
        (and store its square), using the Woodbury identity on the Nyström factors.

        Requires that self.W (s*s) and self.C (n*s) are already set
        via a prior call to unif_nystrom or rec_nystrom. Make sure you ran:
        self.unif_nystrom, self.rec_nystrom or self.full_K before calling this function
        The result is stored in self.sketched_inverse_K_sq (s*s).

        Avoid training small ensemble on large dataset. This will likely result in similar voting vectors and
        increase the chance for the landmark matrix W to be singular. 
        """
        W = self.W
        C = self.C
        lam = self.learning_rate
        q = 1 - self.dropout_rate
        retry_count = 0

        while retry_count < 10:
            try:
                W_inv = np.linalg.pinv(W)
                break
            except np.linalg.LinAlgError:
                retry_count += 1
                if retry_count == 10:
                    raise RuntimeError("SVD did not converge after 10 attempts.")
        CTC = C.T @ C
        try:
            sketched_inverse_K = (W_inv @ C.T @ (lam * np.eye(C.shape[0])) 
                                - lam**2 * (W_inv @ CTC @ np.linalg.inv((1/q) * W + lam * CTC)) @ C.T)
            sketched_inverse_K_sq = sketched_inverse_K @ sketched_inverse_K.T
            self.sketched_inverse_K_sq = sketched_inverse_K_sq

        except np.linalg.LinAlgError:
            print("Singular matrix encountered during sketching. Falling back to full K.")
            K = self.full_K()

    def sketch_r(self, x, vector=False):
        """
        Compute the BRAT weight for a new point x,
        either from the sketched inverse (if available) or exactly.

        Parameters
        ----------
        x : array-like, shape (n_features,)
            A single test point.

        vector : bool, default=False
            If True, also return the raw influence vector (sketched_k or rn)
            along with its norm.

        Returns
        -------
        rn_norm : float
            The clipped norm of the (approximate) influence vector, capped at 10.

        If vector=True, returns a tuple (r_vec, rn_norm), where
        r_vec is the m-dim landmark-space vector (if sketched) or
            the full n-vector rn (if using the exact K).
        """
        if self.sketched_inverse_K_sq is not None:      
            sketched_k = self.sketch_k(x)
            rn_norm = np.sqrt(sketched_k.T @ self.sketched_inverse_K_sq @ sketched_k)
            if rn_norm > 10:
                rn_norm = 1.0
            return rn_norm
        else:
            k = compute_k_vector(self, self.X_train, x)
            rn = np.linalg.pinv((1/self.learning_rate) * self.K + (1-self.dropout_rate) * np.eye(self.K.shape[0])) @ k
            rn_norm = np.linalg.norm(rn)
            if rn_norm > 10:
                rn_norm = 1.0
            if vector:
                return rn, rn_norm
            else:
                return rn_norm
    
    def est_sigma_hat2(self, in_bag):
        """
        Estimate the variance of the noise.
        Parameters
        ----------
        in_bag: Boolean. If True, will use the training set to estimate the variance

        Return
        ------
        sigma_hat2: Estimated variance.
        """
        X_train = self.X_train
        y_train = self.y_train
        X_test = self.X_test
        y_test = self.y_test
        if in_bag:
            n = X_train.shape[0]
            mse = np.mean((y_train - self.predict(X_train)) ** 2)
            sigma_hat2 = mse * (n/ (n - 1))
        else:
            n = X_test.shape[0]
            mse = np.mean((y_test - self.predict(X_test)) ** 2)
            sigma_hat2 = mse * (n/ (n - 1))
        self.sigma_hat2 = sigma_hat2
        return sigma_hat2

    def est_tau_hat2(self, in_bag, Nystrom_subsample, x):
        """
        Estimate the built-in variance tau2.

        Parameters
        ----------
        in_bag: Boolean. Deciding the estimation of sigma2_hat
        Nystrom_subsample: Nystrom subsample rate used to estimate the BRAT weight vector.
        x: A point of interest of making inference.

        Return
        ------
        sigma_hat2: Cache estiamted noise variane
        rn_norm: Cache BRAT weight vector norm
        tau_hat2: Estimated built-in variance.
        """
        sigma_hat2 = self.est_sigma_hat2(in_bag)
        self.unif_nystrom(Nystrom_subsample)
        self.sketch_K()
        rn_norm = self.sketch_r(x)
        s = (1+self.learning_rate * (1-self.dropout_rate)) / self.learning_rate
        tau_hat2 = (s * rn_norm * sigma_hat2)
        return sigma_hat2, rn_norm, tau_hat2