import numpy as np
from sklearn.tree import DecisionTreeRegressor

class SubsampledDecisionTreeRegressor:
    def __init__(self, subsample_rate=0.8, max_depth=None, min_samples_split=2):
        self.subsample_rate = subsample_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = DecisionTreeRegressor(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
        self.subsample = None
        self.leaf_assignments = None

    def fit(self, X, y):
        n_samples = int(np.round(len(y) * self.subsample_rate))
        self.indices = np.random.choice(len(y), n_samples, replace=False)
        self.subsample = np.zeros(len(y), dtype=bool)
        self.subsample[self.indices] = True
        X_subsample = X[self.indices]
        y_subsample = y[self.indices]

        self.tree.fit(X_subsample, y_subsample)
        self.leaf_assignments = self.tree.apply(X)

    def predict(self, X):
        return self.tree.predict(X)
    
    def get_tree(self):
        return self.tree