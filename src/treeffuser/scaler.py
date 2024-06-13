from typing import List
from typing import Optional

from jaxtyping import Float
from numpy import ndarray
from sklearn.preprocessing import StandardScaler


class ScalerMixedTypes:
    """
    Data scaler for mixed data-types with continuous and categorical features.

    Scale continuous features and leave categorical features unchanged. By default, the scaling
    is done using `StandardScaler` from scikit-learn, which standardizes the data to have mean 0
    and standard deviation 1.

    This class does not check the input data. The indices of the categorical features must be
    provided, or they will be treated as continuous and scaled.

    Parameters
    ----------
    scaler : scaler from sklearn.preprocessing, optional
        The scaler to use for continuous features. Default is `StandardScaler` if not provided.
    """

    def __init__(
        self,
        scaler=None,
    ) -> None:
        self._cat_idx = None
        self._is_fitted = False
        self._x_dim = None
        if scaler is None:
            self._scaler = StandardScaler()

    def fit(self, X: Float[ndarray, "batch x_dim"], cat_idx: Optional[List[int]] = None):
        """
        Fit the scaler provided at initialization to the data.

        Parameters
        ----------
        X : ndarray of shape (batch, x_dim)
            The data to fit the scaler to.
        cat_idx : list of int, optional
            The indices of the categorical features.
        """
        self._reset()
        cat_idx = cat_idx if cat_idx is not None else []
        non_cat_idx = [i for i in range(X.shape[1]) if i not in cat_idx]
        X_non_cat = X[:, non_cat_idx]

        self._scaler.fit(X_non_cat)
        self._cat_idx = cat_idx
        self._is_fitted = True
        self._x_dim = X.shape[1]
        return self

    def transform(self, X: Float[ndarray, "batch x_dim"]) -> Float[ndarray, "batch x_dim"]:
        """
        Standardize/scale the data. The categorical features are left unchanged.

        Parameters
        ----------
        X : ndarray of shape (batch, x_dim)
            The data to transform.

        Returns
        -------
        X_transformed : ndarray of shape (batch, x_dim)
            The transformed data with scaled continuous features.
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
        return X

    def fit_transform(
        self, X: Float[ndarray, "batch x_dim"], cat_idx: Optional[List[int]] = None
    ) -> Float[ndarray, "batch x_dim"]:
        """
        Fit the scaler and transform the data in one step.

        Parameters
        ----------
        X : ndarray of shape (batch, x_dim)
            The data to fit and transform.
        cat_idx : list of int, optional
            The indices of the categorical features.

        Returns
        -------
        X_transformed : ndarray of shape (batch, x_dim)
            The transformed data with scaled continuous features.

        See Also
        --------
        fit : Fit the preprocessor to the data.
        transform : Transform the data.
        """
        self.fit(X, cat_idx)
        return self.transform(X)

    def inverse_transform(
        self, X: Float[ndarray, "batch x_dim"]
    ) -> Float[ndarray, "batch x_dim"]:
        """
        Takes the data back to the original scale.

        Parameters
        ----------
        X : ndarray of shape (batch, x_dim)
            The data to transform back.

        Returns
        -------
        X_untransformed : ndarray of shape (batch, x_dim)
            The untransformed data with the original scale.
        """
        if not self._is_fitted:
            raise ValueError("The preprocessor has not been fitted yet.")

        if X.shape[1] != self._x_dim:
            raise ValueError("The input data has a different dimension than the fitted data.")

        X = X.copy()
        non_cat_idx = [i for i in range(X.shape[1]) if i not in self._cat_idx]
        X_non_cat = X[:, non_cat_idx]
        X_non_cat = self._scaler.inverse_transform(X_non_cat)
        X[:, non_cat_idx] = X_non_cat
        return X

    def _reset(self):
        """
        Resets the state of the preprocessor.
        """
        self._scaler = self._scaler.__class__()
        self._cat_idx = None
        self._is_fitted = False
        self._x_dim = None
        return self
