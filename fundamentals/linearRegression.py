# Write linear regression from scratch using numpy

# Y = WX + B

import numpy as np
from logging_config import get_logger
from utils import timeit, normalize_features


class LinearRegression:
    def __init__(self, lr=0.0001):
        self.lr = lr
        self.logger = get_logger(__name__)



    @timeit
    def fit_non_vectorized(self, X: np.ndarray, Y: np.ndarray, nEpochs=100):
        """
        Given X (n_examples, n_features) and Y (n_examples), fit a linear regression model using stochastic gradient descent.
        This is a non-vectorized implementation. Written to highlight the time difference between a vectorized approach and
        a non-vectorized approach.
        """
        normalized = normalize_features(X)
        X = normalized.get("normalized")
        self.mean = normalized.get("mean")
        self.stddev = normalized.get("stddev")
        nFeatures = X.shape[1]
        self.weights = np.random.rand(nFeatures)
        self.bias = 0

        for epoch in range(nEpochs):
            yhat = np.matmul(X, self.weights) + self.bias
            for sample in range(X.shape[0]):
                for feature in range(nFeatures):
                    self.weights[feature] = (
                        self.weights[feature]
                        - self.lr * (yhat[sample] - Y[sample]) * X[sample][feature]
                    )
                self.bias = self.bias - self.lr * (yhat[sample] - Y[sample])
            loss = np.mean(np.sum(Y - yhat) ** 2)
            if epoch % 10 == 0:
                self.logger.debug(f"Epoch: {epoch}, Loss: {loss}")

        self.logger.info(
            f"Trained Parameters are\nWeights: {self.weights}, Bias: {self.bias}"
        )

    @timeit
    def fit(self, X: np.ndarray, Y: np.ndarray, nEpochs: int = 100):
        """
        Given X (n_examples, n_features) and Y (n_examples), fit a linear regression model using stochastic gradient descent
        """
        normalized = normalize_features(X)

        nFeatures = X.shape[1]
        self.weights = np.random.rand(nFeatures)
        self.bias = 0
        self.mean = normalized.get("mean")
        self.stddev = normalized.get("stddev")
        X = normalized.get("normalized")
        for epoch in range(nEpochs):
            yhat = np.matmul(X, self.weights) + self.bias
            for sample in range(X.shape[0]):
                self.weights = (
                    self.weights - self.lr * (yhat[sample] - Y[sample]) * X[sample]
                )
                self.bias = self.bias - self.lr * (yhat[sample] - Y[sample])
            loss = np.mean(np.sum(Y - yhat) ** 2)
            if epoch % 10 == 0:
                self.logger.debug(f"Epoch: {epoch}, Loss: {loss}")
        self.logger.info(
            f"Trained Parameters are\nWeights: {self.weights}, Bias: {self.bias}"
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Given a set of data points, return an array with the predicted outputs
        """
        X = (X - self.mean) / self.stddev
        yhat = np.matmul(X, self.weights) + self.bias
        return yhat


if __name__ == "__main__":
    linearReg = LinearRegression()
    m = 10000
    xvals = np.random.randn(m, 5)
    # print(f"Xvals: {xvals}")
    y = (
        3 * xvals[:, 0]
        + 4 * xvals[:, 1]
        + 4 * xvals[:, 2]
        + 4 * xvals[:, 3]
        + 4 * xvals[:, 4]
        + 5 * np.ones((m))
    )
    # print(y.shape)
    # linearReg.fit_non_vectorized(xvals, y)
    linearReg.fit(xvals, y)
    yhat = linearReg.predict(xvals)
    print(np.linalg.norm(y - yhat))
    # print(xvals)
    # print("----")
    # linearReg._normalize_features(xvals)
