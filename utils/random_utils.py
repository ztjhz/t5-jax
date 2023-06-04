import jax.numpy as jnp
from jax import random


INT32_MIN = jnp.iinfo(jnp.int32).min
INT32_MAX = jnp.iinfo(jnp.int32).max


# referenced from https://github.com/ayaka14732/bart-base-jax/blob/main/lib/random/wrapper.py
def key2seed(key: any) -> int:
    return random.randint(key, (), INT32_MIN, INT32_MAX).item()
