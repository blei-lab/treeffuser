"""
Available methods need to inherit from sklearn.base.BaseEstimator (and maybe more?)
"""

from testbed.data.utils import get_data
from testbed.data.utils import list_data

__all__ = ["get_data", "list_data", "list_methods", "get_method", "list_missing_methods"]
