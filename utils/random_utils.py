import jax.numpy as jnp
from jax import random


UINT16_MIN = jnp.iinfo(jnp.uint16).min
UINT16_MAX = jnp.iinfo(jnp.uint16).max


# referenced from https://github.com/ayaka14732/bart-base-jax/blob/main/lib/random/wrapper.py
def key2seed(key: any) -> int:
    return random.randint(key, (), UINT16_MIN, UINT16_MAX).item()
