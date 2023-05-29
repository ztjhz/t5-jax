from jax import random
import jax.numpy as jnp

from model.dropout import dropout

seed = 2418

key = random.PRNGKey(seed)
x = jnp.ones(shape=(100, 100, 100))
y = dropout(key, x)

assert jnp.allclose(1., y.mean(), atol=1e-3)
