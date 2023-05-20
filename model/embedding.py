import jax.numpy as jnp
from jax import random

from typing import Dict


def init_embedding(num_embeddings: int, embedding_dim: int) -> jnp.ndarray:
    key = random.PRNGKey(2418)
    return random.normal(key, shape=(num_embeddings,embedding_dim))


def fwd_embedding(params: Dict, x: jnp.ndarray) -> jnp.ndarray:
    embedding = params["embedding"]
    y = embedding[x]
    return y
    