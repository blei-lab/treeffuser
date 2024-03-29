"""
No explicit loading for now, to avoid errors if not all models are available.
"""

# import warnings
#
# _AVAILABLE_MODELS = {}
# _MISSING_MODELS = {}
#
# # ########### Import models that are available ###########
# try:
#     from testbed.models.ngboost_ import NGBoostGaussian, NGBoostMixtureGaussian
#
#     _AVAILABLE_MODELS["
#
# except ImportError as e:
#     message = (
#         "Cannot use `ngboost` method. Have you installed `ngboost`? "
#         "You can install it via `pip install ngboost`. "
#         f"The error message was: {e.msg}"
#     )
#     _MISSING_MODELS["ngboost"] = message
#     warnings.warn(message, stacklevel=2)
#
# try:
#     from testbed.models.card import Card
#
#     _AVAILABLE_MODELS["card"] = Card
#
# except ImportError as e:
#     message = (
#         "Cannot use `card` method. Have you installed `lightning-uq-box`? "
#         "You can install it via `pip install lightning-uq-box`. "
#         f"The error message was: {e.msg}"
#     )
#     _MISSING_MODELS["card"] = message
#     warnings.warn(message)
#
# # ########### End of import models that are available ###########
#
#
# def list_models():
#     return list(_AVAILABLE_MODELS.keys())
#
#
# def list_missing_models():
#     return _MISSING_MODELS.copy()
#
#
# def get_model(method_name: str):
#     return _AVAILABLE_MODELS[method_name]
#
#
# __all__ = ["list_models", "list_missing_models", "get_model"] + list_models()
