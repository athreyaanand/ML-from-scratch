import numpy as np

class LinearRegression:
    def __init__(self, X, y, alpha = 0.01, iterations = 1500):
        self.X = X
        self.y = y
        self.alpha = alpha
        self.iterations = iterations

        self.num_samples = len(y)
        self.num_features = X.shape[1]
        # normalize and add bias to X
        self.X = self.normalize_add_bias(self.X, self.num_samples)
        # reshape y
        self.y = y[:, np.newaxis]
        # add extra param to control bias term
        self.params = np.zeros((self.num_features + 1, 1))
    
    def normalize_add_bias(self, X, num_samples):
        # normalize X
        X = (X - np.mean(X, 0)) / np.std(X, 0)
        # add bias term into X itself so we don't have to worry about it
        X = np.hstack((np.ones((num_samples, 1)), X))

        return X

    def fit(self):
        for _ in range(self.iterations):
            # calculate derivative
            d = self.X.T @ (self.X @ self.params - self.y)
            # update function
            self.params = self.params - (self.alpha/self.num_samples) * d
        
        return self

    def score(self, X = None, y = None):
        if X is None:
            X = self.X
        else:
            X = self.normalize_add_bias(X, X.shape[0])

        if y is None:
            y = self.y
        else:
            y = y[:, np.newaxis]
            
        y_pred = X @ self.params
        score = 1 - (((y - y_pred)**2).sum() / ((y - y.mean())**2).sum())

        return score

    def predict(self, X):
        return self.normalize_add_bias(X, X.shape[0]) @ self.params

    def get_params(self):
        return self.params