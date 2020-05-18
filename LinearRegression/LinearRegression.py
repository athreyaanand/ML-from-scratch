import numpy as np

class LinearRegression:
    def __init__(self, X, y, alpha = 0.01, iterations = 1500):
        self.X = X
        self.y = y
        self.alpha = alpha
        self.iterations = iterations

        self.num_samples = len(y)
        self.num_features = X.shape[1]
        # normalize X
        X = (X - np.mean(X, 0)) / np.std(X, 0)
        # add bias term into X itself so we don't have to worry about it
        self.X = np.hstack((np.ones(self.num_samples, 1)), X)
        #reshape y
        self.y = y[:, np.newaxis]
        self.params = np.zeros((self.num_features + 1, 1))

    def fit(self):
        for _ in range(self.iterations):
            # calculate derivative
            d = self.X.T @ (self.X @ self.params - self.y)
            # update function
            self.params = self.params - (self.alpha/self.num_samples) * d
        
        return self

    def score(self):
        X = self.X
        y = self.y
        # TODO: add functionality to input different X and y from init
        y_pred = X @ self.params
        score = 1 - (((y - y_pred)**2).sum() / ((y - y.mean())**2).sum())

        return score

    def predict(self, X):
        # normalize X
        X = (X - np.mean(X, 0)) / np.std(X, 0)
        # add bias term
        X = np.hstack((np.ones((X.shape[0], 1)), X))

        return X @ self.params

    def get_params(self):
        return self.params