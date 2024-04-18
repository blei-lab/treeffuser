"""
Helper functions to evaluate predictions and their divergence from a boosted forest from Booster.dump_model().
"""

from typing import List

import numpy as np
from jaxtyping import Float


def _predict(
    forest: List[dict],
    y: Float[np.ndarray, "y_dim"],
    x: Float[np.ndarray, "x_dim"],
    t: float,
):
    score = 0
    for tree in forest:
        leaf = _get_leaf(tree["tree_structure"], y, x, t)
        score += _predict_leaf(leaf, y, x, t)
    return score


def _predict_leaf(
    node: dict,
    y: Float[np.ndarray, "y_dim"],
    x: Float[np.ndarray, "x_dim"],
    t: float,
):
    y_dim = len(y)
    n_features = y_dim + len(x) + 1

    score = 0
    for feature_index, feature_coef in zip(node["leaf_features"], node["leaf_coeff"]):
        if feature_index < y_dim:
            feature_value = y[feature_index]
        elif feature_index == n_features - 1:
            feature_value = t
        else:
            feature_value = x[feature_index - y_dim]
        score += feature_coef * feature_value
    return node["leaf_const"] + score


def _compute_prediction_divergence(
    forest: List[dict],
    y: Float[np.ndarray, "y_dim"],
    x: Float[np.ndarray, "x_dim"],
    t: float,
):
    y_dim = len(y)
    div = 0
    for tree in forest:
        leaf = _get_leaf(tree["tree_structure"], y, x, t)
        div += _compute_prediction_divergence_leaf(leaf, y_dim)
    return div


def _compute_prediction_divergence_leaf(node: dict, y_dim: int):
    div = 0
    for feature_index, feature_coef in zip(node["leaf_features"], node["leaf_coeff"]):
        if feature_index < y_dim:
            div += feature_coef
    return div


def _get_leaf(
    node: dict,
    y: Float[np.ndarray, "y_dim"],
    x: Float[np.ndarray, "x_dim"],
    t: float,
):
    """
    Traverse a tree recursively to return the leaf node corresponding to input features (y, x, t).

    Parameters
    ----------
    node : dict
        The current node in the decision tree.
    y : Float[np.ndarray, "y_dim"]
        The 'y' features as a numpy array.
    x : Float[np.ndarray, "x_dim"]
        The 'x' features as a numpy array.
    t : float
        The time feature.

    Notes
    -----
    Assumes that the node is a dictionary compatible with Booster.dump_model()[0]["tree_structure"].
    Assumes that the splititng features are ordered as (y, x, t).
    """
    y_dim = len(y)
    n_features = y_dim + len(x) + 1

    if "leaf_index" in node:
        return node

    threshold = node["threshold"]
    split_feature = int(node["split_feature"])
    if split_feature < y_dim:  # split on y
        go_left = _check_split(y[split_feature], threshold, node["decision_type"])
    elif split_feature == n_features - 1:  # split on t
        go_left = _check_split(t, threshold, node["decision_type"])
    else:  # split on x
        split_feature -= y_dim
        go_left = _check_split(x[split_feature], threshold, node["decision_type"])

    return (
        _get_leaf(node["left_child"], y, x, t)
        if go_left
        else _get_leaf(node["right_child"], y, x, t)
    )


def _check_split(value: float, threshold: float, decision_type: str):
    if decision_type == "<=":
        return value <= threshold
    elif decision_type == "==":  # used for categorical variables
        return value == threshold
    else:
        raise NotImplementedError


def _has_feature(node: dict, target_feature: int = 0):
    if "leaf_index" not in node:
        return False

    if node["split_feature"] == target_feature:
        return True

    left_check = _has_feature(node["left_child"], target_feature)
    right_check = _has_feature(node["right_child"], target_feature)

    return left_check or right_check


######################################################## TRASH ########################################################


# def _compute_score_divergence_old(
#     forest: List[dict],
#     learning_rate: float,
#     y: Float[np.ndarray, "y_dim"],
#     x: Float[np.ndarray, "x_dim"],
#     t: float,
# ):
#     div = 0
#     for tree in forest:
#         div += _compute_divergence_tree_old(tree["tree_structure"], y, x, t)
#     return learning_rate * div


# def _compute_divergence_tree_old(
#     node: dict,
#     y: Float[np.ndarray, "y_dim"],
#     x: Float[np.ndarray, "x_dim"],
#     t: float,
# ):
#     """
#     Assumes that the spliiting features y, x, and t follow this order: (y, x, t).
#     """
#     y_dim = len(y)
#     n_features = y_dim + len(x) + 1

#     if "leaf_index" in node:
#         return _compute_divergence_leaf(node, y_dim)

#     threshold = node["threshold"]
#     split_feature = int(node["split_feature"])
#     if split_feature < y_dim:  # split on y
#         go_left = _check_split(y[split_feature], threshold, node["decision_type"])
#     elif split_feature == n_features - 1:  # split on t
#         go_left = _check_split(t, threshold, node["decision_type"])
#     else:  # split on x
#         split_feature -= y_dim
#         go_left = _check_split(x[split_feature], threshold, node["decision_type"])

#     return (
#         _compute_divergence_tree(node["left_child"], y, x, t)
#         if go_left
#         else _compute_divergence_tree(node["right_child"], y, x, t)
#     )

#######################################################################################################################

# def _integrate_divergence_over_time(
#     forest: List[dict],
#     learning_rate: float,
#     y: Float[np.ndarray, "y_dim"],
#     x: Float[np.ndarray, "x_dim"],
#     T: float=1,
# ):
#     integral = 0
#     for tree in forest:
#         integral += _integrate_divergence_tree_over_time(
#             tree["tree_structure"], y, x, t_min=0, t_max=T
#         )
#     return learning_rate * integral


# def _integrate_divergence_tree_over_time(
#     node: dict,
#     y: Float[np.ndarray, "y_dim"],
#     x: Float[np.ndarray, "x_dim"],
#     t_min: float,
#     t_max: float,
# ):
#     """
#     Assumes that the spliiting features y, x, and t follow this order: (y, x, t).
#     """
#     y_dim = len(y)
#     n_features = y_dim + len(x) + 1

#     if "leaf_index" in node:
#         return _integrate_divergence_leaf_over_time(node, y_dim, t_min, t_max)

#     threshold = node["threshold"]
#     split_feature = int(node["split_feature"])
#     t_min_left = t_min_right = t_min
#     t_max_left = t_max_right = t_max
#     if split_feature == n_features - 1:  # split on t
#         integrate_left, integrate_right = True, True
#         t_max_left = t_min_right = threshold
#     elif split_feature < y_dim:  # split on y
#         integrate_left = _check_split(y[split_feature], threshold, node["decision_type"])
#         integrate_right = not integrate_left
#     else:  # split on x
#         split_feature -= y_dim
#         integrate_left = _check_split(x[split_feature], threshold, node["decision_type"])
#         integrate_right = not integrate_left

#     integral_left = (
#         _integrate_divergence_tree_over_time(node["left_child"], y, x, t_min_left, t_max_left)
#         if integrate_left
#         else 0
#     )

#     integral_right = (
#         _integrate_divergence_tree_over_time(
#             node["right_child"], y, x, t_min_right, t_max_right
#         )
#         if integrate_right
#         else 0
#     )

#     return integral_left + integral_right


# def _check_split(value: float, threshold: float, decision_type: str):
#     if decision_type == "<=":
#         return value <= threshold
#     elif decision_type == "==":
#         return value == threshold
#     else:
#         raise NotImplementedError


# def _integrate_divergence_leaf_over_time(node: dict, y_dim: int, t_min, t_max):
#     out = 0
#     for feature_index, feature_coef in zip(node["leaf_features"], node["leaf_coeff"]):
#         if feature_index <= y_dim:
#             out += feature_coef
#     return out * (t_max - t_min)
