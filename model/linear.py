from typing import Dict, List

import jax.numpy as jnp


def fwd_linear(params: Dict, x: jnp.ndarray) -> jnp.ndarray:
    kernel = params['kernel']
    bias = params.get('bias', None)

    y = jnp.dot(x, kernel)
    if bias is not None: y += bias

    return y
