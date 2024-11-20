import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier

class CustomerPredictionModel:
    def __init__(self, n_trees=100, max_depth=5, cost_sensitive=False):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.decision_trees = []
        self.cost_sensitive = cost_sensitive
        self.gbm_model = None

    def calculate_adaptive_threshold(self, y):
        high_risk_count = np.sum(y == 1)
        low_risk_count = np.sum(y == 0)
        total_count = len(y)

        if total_count == 0:
            return 0.5  # Return a neutral threshold if no data

        high_risk_ratio = high_risk_count / total_count
        low_risk_ratio = low_risk_count / total_count

        min_threshold = 0.1
        max_threshold = 0.9

        dynamic_threshold = (
            (high_risk_ratio * 0.9) +
            (low_risk_ratio * 0.1) +
            0.5 * (1 - high_risk_ratio)
        )

        return max(min_threshold, min(max_threshold, dynamic_threshold))

    def calculate_sample_weights(self, y):
        high_risk_count = np.sum(y == 1)
        low_risk_count = len(y) - high_risk_count
        weight_high_risk = (low_risk_count / (high_risk_count)) * 1.5
        weight_low_risk = 1

        sample_weights = np.where(y == 1, weight_high_risk, weight_low_risk)
        return sample_weights

    def fit(self, X, y):
        self.base_threshold = self.calculate_adaptive_threshold(y)
        sample_weights = self.calculate_sample_weights(y)

        for _ in range(self.n_trees):
            while True:
                bootstrap_indices = np.random.choice(len(X), len(X), replace=True, p=sample_weights / np.sum(sample_weights))
                y_bootstrap = y[bootstrap_indices]
                if np.unique(y_bootstrap).size > 1:
                    break

            X_bootstrap = X[bootstrap_indices]
            sample_weights_bootstrap = sample_weights[bootstrap_indices]

            tree = DecisionTreeClassifier(max_depth=self.max_depth)
            tree.fit(X_bootstrap, y_bootstrap, sample_weight=sample_weights_bootstrap)
            self.decision_trees.append(tree)

        if self.cost_sensitive:
            self.gbm_model = GradientBoostingClassifier(n_estimators=self.n_trees, max_depth=self.max_depth, learning_rate=0.1)
            self.gbm_model.fit(X, y, sample_weight=sample_weights)

    def predict(self, X):
        tree_predictions = np.array([tree.predict(X) for tree in self.decision_trees])
        majority_votes = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=tree_predictions)

        if self.cost_sensitive and self.gbm_model is not None:
            gbm_proba = self.gbm_model.predict_proba(X)[:, 1]
            gbm_predictions = (gbm_proba >= self.base_threshold - 0.04).astype(int)
            ensemble_predictions = np.round((0.4 * majority_votes + 0.6 * gbm_predictions)).astype(int)
            return ensemble_predictions
        else:
            return majority_votes
