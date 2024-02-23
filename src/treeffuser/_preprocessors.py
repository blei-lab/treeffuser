from typing import List
from typing import Optional

from jaxtyping import Float
from numpy import ndarray
from sklearn.preprocessing import StandardScaler


class Preprocessor:
    """
    General lightweight preprocessor for data.

    In addition to handling simple stuff like normalization and standardization
    of continuous data we do some lightweight standardization of categorical data.

    """

    def __init__(
        self,
    ) -> None:
        self._scaler = None
        self._cat_idx = None
        self._is_fitted = False
        self._x_dim = None

    def fit(self, X: Float[ndarray, "batch x_dim"], cat_idx: Optional[List[int]] = None):
        """
        Standardizes the data to have mean 0 and standard deviation 1.

        This class does no checking of the input data. It is assumed that the user
        has already checked that the input data is valid.

        Args:
        X: The data to fit the preprocessor to.
        cat_idx: The indices of the categorical features.
        """
        self._reset()

        X_non_cat = X[:, [i for i in range(X.shape[1]) if i not in cat_idx]]

        self._scaler = StandardScaler()
        self._scaler.fit(X_non_cat)
        self._cat_idx = cat_idx
        self._is_fitted = True
        self._x_dim = X.shape[1]
        return self

    def transform(self, X: Float[ndarray, "batch x_dim"]) -> Float[ndarray, "batch x_dim"]:
        """
        Standardizes the data to have mean 0 and standard deviation 1. The categorical
        features are left unchanged.

        This class does no checking of the input data. It is assumed that the user
        has already checked that the input data is valid.

        Args:
        X: The data to transform.
        """
        if not self._is_fitted:
            raise ValueError("The preprocessor has not been fitted yet.")

        if X.shape[1] != self._x_dim:
            raise ValueError("The input data has a different dimension than the fitted data.")

        X = X.copy()
        non_cat_idx = [i for i in range(X.shape[1]) if i not in self._cat_idx]
        X_non_cat = X[:, non_cat_idx]
        X_non_cat = self._scaler.transform(X_non_cat)
        X[:, non_cat_idx] = X_non_cat
        return self.scaler.transform(X)

    def fit_transform(
        self, X: Float[ndarray, "batch x_dim"], cat_idx: Optional[List[int]] = None
    ) -> Float[ndarray, "batch x_dim"]:
        """
        Standardizes the data to have mean 0 and standard deviation 1.

        This class does no checking of the input data. It is assumed that the user
        has already checked that the input data is valid.

        Args:
        X: The data to fit the preprocessor to.
        cat_idx: The indices of the categorical features.
        """
        self.fit(X, cat_idx)
        return self.transform(X)

    def _reset(self):
        """
        Resets the state of the preprocessor.
        """
        self._scaler = None
        self._cat_idx = None
        self._is_fitted = False
        return self