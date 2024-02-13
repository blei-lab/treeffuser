"""
This should be the main file corresponding to the project.
"""

from sklearn.base import BaseEstimator


class Treeffuser(BaseEstimator):

    def __init__(self, *args, **kwargs):
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass

    def sample(self, X):
        pass

    def likelihood(self, X, y):
        """
        Something that computes the log-likelihood of the model.
        """

    def pred_distribution(self, X):
        """
        Maybe the CDF?
        """
