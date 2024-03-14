"""
Available methods need to inherit from the BaseEstimators
"""

_AVAILABLE_METHODS = dict()
_MISSING_METHODS = dict()

try:
    from _ngboost import NGBRegressor

    _AVAILABLE_METHODS["ngboost"] = NGBRegressor

except ImportError:
    _MISSING_METHODS["ngboost"] = (
        "Install `ngboost` to use this method: `pip install ngboost`."
    )


def list_methods():
    return list(_AVAILABLE_METHODS.keys())


def list_missing_methods():
    return _MISSING_METHODS.copy()


def get_method(method_name: str):
    return _AVAILABLE_METHODS[method_name]


from data.utils import get_data
from data.utils import list_data

__all__ = ["get_data", "list_data", "list_methods", "get_method", "list_missing_methods"]
