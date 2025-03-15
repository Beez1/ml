import numpy as np

class LinearRegressionGD:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        """
        Train the model using Gradient Descent.
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.epochs):
            y_pred = np.dot(X, self.weights) + self.bias
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)
            
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict(self, X):
        if self.weights is None or self.bias is None:
            raise ValueError("Model has not been trained yet. Call `fit()` before `predict()`.")
        return np.dot(X, self.weights) + self.bias

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