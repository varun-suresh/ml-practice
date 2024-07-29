import numpy as np
from utils import timeit, normalize_features
from logging_config import get_logger

"""
z = w.x + b
f(z) = 1/(1+e^-z)
where f(z) is the logistic regression function 
"""


class LogisticRegression:
    def __init__(self, lr=0.0001):
        self.lr = lr
        self.logger = get_logger(__name__)

    def validate_fit_inputs(self, X: np.ndarray, Y: np.ndarray):
        """
        Check that X and Y have the same number of rows.
        """
        if len(X) != len(Y):
            self.logger.error("Features and labels don't have the same number of rows")

    def validate_predict_inputs(self, X):
        try:
            if len(self.weights) != len(X[0]):
                self.logger.error(
                    f"Number of features in trained model ({len(self.weights)}) don't match the  features in predict input ({len(X[0])})"
                )
        except NameError as e:
            self.logger.error(e)
            self.logger.error("Need to use the fit function first to train the model.")

    @timeit
    def fit(self, X: np.ndarray, Y: np.ndarray, nEpochs: int = 100):
        """
        Function to fit a binary classifier using logistic regression loss
        """
        self.validate_fit_inputs(X, Y)
        normalized = normalize_features(X)
        self.mean = normalized.get("mean")
        self.stddev = normalized.get("stddev")
        X = normalized.get("normalized")
        nSamples = X.shape[0]
        self.weights = np.random.rand(nFeatures)
        self.bias = 0
        for epoch in range(nEpochs):
            loss = 0
            logits = X @ self.weights + self.bias
            yhat = 1 / (1 + np.exp(-logits))

            for i in range(nSamples):
                self.weights = self.weights - self.lr * (yhat[i] - Y[i]) * X[i]
                self.bias = self.bias - self.lr * (yhat[i] - Y[i])
                loss += -Y[i] * np.log(yhat[i]) - (1 - Y[i]) * np.log(1 - yhat[i])

            if epoch % 10 == 0:
                self.logger.debug(f"Epoch: {epoch}, Loss: {loss}")
        self.logger.info(
            f"Trained Parameters are\nWeights: {self.weights}, Bias: {self.bias}"
        )

    def predict(self, X: np.ndarray):
        """
        Predict using the trained weights and bias
        """
        self.validate_predict_inputs(X)
        normalized_vals = normalize_features(X, self.mean, self.stddev)
        normalized = normalized_vals.get("normalized")
        logits = normalized @ self.weights + self.bias
        yhat = 1 / (1 + np.exp(-logits))
        return yhat


if __name__ == "__main__":
    logreg = LogisticRegression()
    nSamples = 5000
    nFeatures = 2

    X = np.zeros((nSamples, nFeatures))
    X[0 : nSamples // 2, 0:nFeatures] = np.random.rand(nSamples // 2, nFeatures) + 4
    X[nSamples // 2 :, 0:nFeatures] = np.random.rand(nSamples // 2, nFeatures)
    Y = np.zeros((nSamples, 1))
    Y[0 : nSamples // 2] = np.ones((nSamples // 2, 1))

    logreg.fit(X, Y)
    X = np.random.rand(nSamples, nFeatures)
    Y = np.zeros((nSamples, 1))
    yhat = logreg.predict(X)
    error = 0
    for i in range(len(yhat)):
        error += yhat[i] - Y[i]
    print(error)
