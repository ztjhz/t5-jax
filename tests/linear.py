import flax.linen
import jax.numpy as jnp
from jax import random

from model.linear import fwd_linear

# set up random keys
seed = 2418
key = random.PRNGKey(seed)
keys = random.split(key, 3)

# set up params, x
kernel = random.uniform(keys[0], shape=(10, ))
bias = random.uniform(keys[1], shape=(10, ))
x = random.uniform(keys[2], shape=(3, 4, 10))

params = {"kernel": kernel, "bias": bias}
params_flax = {"params": params}

# my linear output
output = fwd_linear(params, x)

# flax linear output
model = flax.linen.linear.Dense(features=x)
output_flax = model.apply(params_flax, x)

assert jnp.allclose(output, output_flax)