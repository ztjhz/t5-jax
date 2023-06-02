from typing import List

import jax.numpy as jnp
from jax import random


def dropout(key: List, x: jnp.ndarray, dropout_rate: int = 0.1) -> jnp.ndarray:
    assert 0. <= dropout_rate <= 1, "dropout_rate should be between 0 and 1"
    keep_rate = 1 - dropout_rate

    # Expected value of y remains the same; E((x * keep_rate) / keep_rate)
    y = x * random.bernoulli(key=key, p=keep_rate, shape=x.shape) / keep_rate

    return y
