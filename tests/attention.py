import flax.linen
import jax.numpy as jnp
from jax import random

from ..model.attention import fwd_attention
from ..utils.attention_utils import split_projection_to_heads

# set up random keys
seed = 2418
key = random.PRNGKey(seed)
keys = random.split(key, 10)

# set up params
batch_size = 4
seq_len = 6
n_head = 12
d_model = 10
d_k = 8
d_v = 8
d_out = d_model

qry_states = random.uniform(keys[0], (batch_size, seq_len, d_model))
tgt_states = random.uniform(keys[1], (batch_size, seq_len, d_model))

q_kernel = random.uniform(keys[2], (d_model, n_head * d_k))
k_kernel = random.uniform(keys[3], (d_model, n_head * d_k))
v_kernel = random.uniform(keys[4], (d_model, n_head * d_v))

q_bias = random.uniform(keys[5], (n_head * d_k,))
k_bias = random.uniform(keys[6], (n_head * d_k,))
v_bias = random.uniform(keys[7], (n_head * d_v,))

out_kernel = random.uniform(keys[8], (n_head * d_v, d_out))
out_kernel_flax = jnp.reshape(out_kernel, (n_head, d_v, d_out)) # has multi-head attention
out_bias = random.uniform(keys[9], (d_out,))

mask_dec_1d = jnp.ones((batch_size, seq_len), dtype=jnp.bool_)
mask = jnp.einsum("bi,bj->bij", mask_dec_1d, mask_dec_1d)[:, None]

params = {
    "q": {"kernel": q_kernel, "bias": q_bias},
    "k": {"kernel": k_kernel, "bias": k_bias},
    "v": {"kernel": v_kernel, "bias": v_bias},
    "o": {"kernel": out_kernel, "bias": out_bias},
}
split_projection_to_heads({"kernel": out_kernel}, n_head)

params_flax = {
    "params": {
        "query": {
            **split_projection_to_heads({"kernel": q_kernel, "bias": q_bias}, n_head)
        },
        "key": {
            **split_projection_to_heads({"kernel": k_kernel, "bias": k_bias}, n_head)
        },
        "value": {
            **split_projection_to_heads({"kernel": v_kernel, "bias": v_bias}, n_head)
        },
        "out": {"kernel": out_kernel_flax, "bias": out_bias},
    }
}

# my output
output = fwd_attention(params, qry_states, tgt_states, mask)

# flax attention output
model = flax.linen.MultiHeadDotProductAttention(
    num_heads=n_head,
    qkv_features=d_k * n_head,
    out_features=d_out,
    broadcast_dropout=False,
)

output_flax = model.apply(params_flax, qry_states, tgt_states, mask=mask)

assert jnp.allclose(output, output_flax)
