import numpy as np

class LinearRegressionGD:
    def __init__(self, learning_rate=0.01, n_iter=100):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.costs = []  # Store the cost function values per epoch

    def fit(self, X, y):
        self.theta = np.zeros(X.shape[1])
        m = len(y)

        for _ in range(self.n_iter):
            y_pred = X.dot(self.theta)  # Compute predictions
            error = y_pred - y
            cost = (1 / (2 * m)) * np.sum(error ** 2)  # Compute cost function
            self.costs.append(cost)  # Append cost for plotting

            gradient = (1 / m) * X.T.dot(error)  # Compute gradient
            self.theta -= self.learning_rate * gradient  # Update theta
    def predict(self, X):
        return np.dot(X, self.theta)

    def mse(self, y_true, y_pred):
        """
        Compute Mean Squared Error (MSE).
        Parameters:
        - y_true: Actual values
        - y_pred: Predicted values
        Returns:
        - Mean Squared Error value
        """
        return np.mean((y_true - y_pred) ** 2)

__all__ = ["linearRegressionGD"]