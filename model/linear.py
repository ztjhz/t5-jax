from typing import Dict

import jax.nn as jnn
import jax.numpy as jnp
from jax import random


def fwd_linear(params: Dict, x: jnp.ndarray) -> jnp.ndarray:
    kernel = params["kernel"]
    bias = params.get("bias", None)

    y = jnp.dot(x, kernel)
    if bias is not None:
        y += bias

    return y


def init_linear(features: int, use_bias: bool = False, seed: int = 2418):
    initializer = jnn.initializers.normal(1)
    key = random.PRNGKey(seed)
    params = {"kernel": initializer(key, (features,))}
    if use_bias:
        params["bias"] = jnp.zeros((features,))
    return params
