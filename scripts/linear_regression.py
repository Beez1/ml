import numpy as np

class LinearRegressionGD:
    def __init__(self, learning_rate=0.01, n_iter=5000, tol=1e-6):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.tol = tol  # Convergence tolerance
        self.costs = []  # Store cost function values

    def fit(self, X, y):
        """
        Train the linear regression model using gradient descent.
        """
        X_bias = np.c_[np.ones((X.shape[0], 1)), X]  # Add bias term
        self.theta = np.zeros(X_bias.shape[1])
        m = len(y)

        for _ in range(self.n_iter):
            y_pred = X_bias.dot(self.theta)  # Compute predictions
            error = y_pred - y
            cost = (1 / (2 * m)) * np.sum(error ** 2)  # Compute cost function
            self.costs.append(cost)  

            gradient = (1 / m) * X_bias.T.dot(error)  # Compute gradient
            self.theta -= self.learning_rate * gradient  # Update weights

            # Convergence Check
            if len(self.costs) > 1 and abs(self.costs[-1] - self.costs[-2]) < self.tol:
                print(f"Converged at iteration {_}")
                break

    def predict(self, X):
        """
        Predict target values using trained model.
        """
        X_bias = np.c_[np.ones((X.shape[0], 1)), X]  # Add bias term
        return np.dot(X_bias, self.theta)

    def mse(self, y_true, y_pred):
        """
        Compute Mean Squared Error (MSE).
        """
        return np.mean((y_true - y_pred) ** 2)

__all__ = ["LinearRegressionGD"]
