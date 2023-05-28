import jax.numpy as jnp

from typing import Dict


# Layer normalization: https://arxiv.org/abs/1607.06450
def fwd_layer_norm(params: Dict, x: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
    weight = params["weight"]
    bias = params.get("bias", None)

    mean = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)

    y = (x - mean) / jnp.sqrt(var + eps) * weight
    if bias is not None:
        y = y + bias

    return y


# Root Mean Square Layer Normalization: https://arxiv.org/abs/1910.07467
# Does not subtract mean and does not have bias
def fwd_layer_norm_rms(params: Dict, x: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
    weight = params["weight"]
    var = x.var(-1, keepdims=True)
    y = x / jnp.sqrt(var + eps) * weight

    return y
