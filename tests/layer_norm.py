import flax.linen
from ..model.layer_norm import fwd_layer_norm

from jax import random
import jax.numpy as jnp

# set up random keys
seed = 2418
key = random.PRNGKey(seed)
keys = random.split(key, 3)

# set up params, x and epsilon
eps = 1e-5
weight = random.uniform(keys[0], shape=(10, ))
bias = random.uniform(keys[1], shape=(10, ))
x = random.uniform(keys[2], shape=(3, 4, 10))

params = {"weight": weight, "bias": bias}
params_flax = {"params": {"scale": weight, "bias": bias}}

# my linear norm output
output = fwd_layer_norm(params, x, eps)

# flax linear norm output
model = flax.linen.LayerNorm(epsilon=eps)
output_flax = model.apply(params_flax, x)

assert jnp.allclose(output, output_flax)