import numpy as np
import torch as t
from jaxtyping import Float


def _to_tensor(X: Float[np.ndarray, "batch x_dim"]) -> Float[t.Tensor, "batch x_dim"]:
    return t.tensor(X, dtype=t.float32)
