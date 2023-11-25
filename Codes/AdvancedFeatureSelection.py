import numpy as np
import cvxpy as cp
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
class AdvancedFeatureSelection:
    def __init__(self):
        pass

    def initialize_feature_selection_vector(self, num_features):
        return np.ones(num_features)

    def discriminative_power(self, X, C, y):
        # Implementation of J1(F)
        #    Parameters:
        # - X: Stacked feature matrix with dimensions N x D, where N is the number of samples and D is the total number of features.
        # - y: Target vector containing class labels for each sample.
        # - C: Total number of classes in the multiclass problem.
        # Returns:
        # - Discriminative power measure J1(F)
        epsilon = 1e-6  # Small constant to prevent division by zero

        # Get the number of samples (N) and features (D)
        N, D = X.shape

        # Initialize discriminative power measure
        discriminative_power_measure = 0.0

        # Iterate over all classes
        for c in range(C):
            # Get indices of samples belonging to class c
            class_indices = np.where(y == c)[0]

            # Calculate means and variances for class c and the rest
            mean_c = np.mean(X[class_indices, :], axis=0)
            variance_c = np.var(X[class_indices, :], axis=0)

            mean_rest = np.mean(X[np.setdiff1d(range(N), class_indices), :], axis=0)
            variance_rest = np.var(X[np.setdiff1d(range(N), class_indices), :], axis=0)

            # Update discriminative power measure using the formula
            discriminative_power_measure += np.sum((mean_c - mean_rest)**2 / (variance_c + variance_rest + epsilon))

        # Normalize the discriminative power measure
        discriminative_power_measure /= C

        return discriminative_power_measure

    def redundancy(self, X, F):
        # Implementation of J2(F)
        D = X.shape[1]
        epsilon = 1e-6 
        # Initialize redundancy measure
        redundancy_measure = 0.0
        # Iterate over all pairs of selected features (i, j) where i < j
        for i in range(D):
            for j in range(i + 1, D):
                # Calculate Pearson correlation coefficient between features i and j
                correlation_coefficient = np.corrcoef(X[:, i], X[:, j])[0, 1]
                # Update redundancy measure using the formula
                redundancy_measure += correlation_coefficient * F[i] * F[j]

        # Normalize the redundancy measure
        redundancy_measure *= 2 / (D * (D - 1) + epsilon)
        return redundancy_measure

    def l1_regularization(self, F):
        # Implementation of L1 regularization
        return np.sum(F)

    def objective_function(self, X, y, F, lmbda, epsilon):
        J1 = self.discriminative_power(X, y)
        J2 = self.redundancy(X, F)
        regularization_term = lmbda * self.l1_regularization(F)
        return J1 / (J2 + epsilon) - regularization_term

    def optimize_feature_selection(self, X, y, lmbda, epsilon, F):
        num_features = X.shape[1]
        F_opt = cp.Variable(num_features, boolean=True)
        objective = cp.Maximize(self.objective_function(X, y, F_opt, lmbda, epsilon))
        constraints = [F_opt[i] == F[i] for i in range(num_features)]
        problem = cp.Problem(objective, constraints)
        problem.solve()
        return F_opt.value

    def cross_validation(self, X, y, F, lmbda, epsilon, n_folds):
        # Implementation of cross-validation
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        performance_metrics = []

        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Apply feature selection to the training and test sets
            X_train_selected = X_train[:, F.astype(bool)]
            X_test_selected = X_test[:, F.astype(bool)]

            rf_classifier = RandomForestClassifier(random_state=42)
            rf_classifier.fit(X_train_selected, y_train)

            # Make predictions on the test set
            y_pred = rf_classifier.predict(X_test_selected)

            # Evaluate performance using accuracy
            accuracy = accuracy_score(y_test, y_pred)
            performance_metrics.append(accuracy)

        return performance_metrics

    def feature_selection(self, X, y, lmbda, epsilon, n_folds):
        F = self.initialize_feature_selection_vector(X.shape[1])
        F_optimized = self.optimize_feature_selection(X, y, lmbda, epsilon, F)
        performance_metrics = self.cross_validation(X, y, F_optimized, lmbda, epsilon, n_folds)
        return F_optimized, performance_metrics


