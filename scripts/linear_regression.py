import numpy as np

class LinearRegressionGD:
    def __init__(self, learning_rate=0.01, n_iter=5000, tol=1e-6):
        # Set up the basic parameters we need for training
        self.learning_rate = learning_rate  # How big of steps we take during training
        self.n_iter = n_iter  # How many times we'll go through the training data
        self.tol = tol  # Stop training if we're not improving much
        self.costs = []  # Keep track of how well we're doing during training

    def fit(self, X, y):
        """
        Train the linear regression model using gradient descent.
        
        Args:
            X: Input features matrix
            y: Target values vector
        """
        # Add a column of 1s to handle the bias term
        X_bias = np.c_[np.ones((X.shape[0], 1)), X]
        
        # Start with all weights set to zero
        self.theta = np.zeros(X_bias.shape[1])
        m = len(y)  # How many training examples we have

        # Keep training until we hit our max iterations
        for _ in range(self.n_iter):
            # Make predictions with current weights
            y_pred = X_bias.dot(self.theta)
            
            # See how far off our predictions are
            error = y_pred - y
            
            # Calculate how badly we're doing (mean squared error)
            cost = (1 / (2 * m)) * np.sum(error ** 2)
            self.costs.append(cost)  

            # Figure out which direction to adjust our weights
            gradient = (1 / m) * X_bias.T.dot(error)
            
            # Update our weights - move a bit in the opposite direction of the gradient
            self.theta -= self.learning_rate * gradient

            # If we're not improving much anymore, we can stop early
            if len(self.costs) > 1 and abs(self.costs[-1] - self.costs[-2]) < self.tol:
                print(f"Converged at iteration {_}")
                break

    def predict(self, X):
        """
        Predict target values using trained model.
        
        Args:
            X: Input features matrix
            
        Returns:
            Predicted values vector
        """
        # Add the bias term and multiply by our trained weights
        X_bias = np.c_[np.ones((X.shape[0], 1)), X]
        return np.dot(X_bias, self.theta)

    def mse(self, y_true, y_pred):
        """
        Compute Mean Squared Error (MSE).
        
        Args:
            y_true: Actual target values
            y_pred: Predicted values
            
        Returns:
            Mean squared error value
        """
        # Calculate average squared difference between true and predicted values
        return np.mean((y_true - y_pred) ** 2)

__all__ = ["LinearRegressionGD"]