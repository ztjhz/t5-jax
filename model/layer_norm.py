import jax.numpy as jnp

from typing import Dict


# Layer normalization: https://arxiv.org/abs/1607.06450
def fwd_layer_norm(params: Dict,
                   x: jnp.ndarray,
                   eps: float = 1e-5) -> jnp.ndarray:
    weight = params['weight']
    bias = params.get('bias', None)

    mean = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)

    y = (x - mean) / jnp.sqrt(var + eps) * weight
    if bias is not None: y = y + bias

    return y