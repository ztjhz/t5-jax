from typing import Dict, List

import jax.numpy as jnp
from jax import random
import jax.nn as nn

from ..model.attention import fwd_attention
from ..model.layer_norm import fwd_layer_norm
from ..model.dropout import dropout
from ..model.linear import fwd_linear


def fwd_transformer_encoder(
    params: Dict, qry_states: List[float], mask: List[bool], dropout_key: List = None
):
    self_attn_block, ff_block = params["0"], params["1"]

    self_attn = self_attn_block["SelfAttention"]
    self_attn_layer_norm = self_attn_block["layer_norm"]

    ff_0 = ff_block["DenseReluDense"]["wi"]
    ff_1 = ff_block["DenseReluDense"]["wo"]
    ff_layer_norm = ff_block["layer_norm"]

    if dropout_key is not None:
        subkeys = random.split(dropout_key, 3)

    # Pre layer norm
    normed_qry_states = fwd_layer_norm(self_attn_layer_norm, qry_states)

    # Multi-head attention
    x = fwd_attention(self_attn, normed_qry_states, normed_qry_states, mask)

    # Dropout
    if dropout_key is not None:
        x = dropout(subkeys[0], x)

    # Add
    x = x + qry_states

    # Feed Forward (Linear -> Gelu -> Dropout -> Linear -> Dropout)
    _x = x

    # Pre layer norm
    x = fwd_layer_norm(ff_layer_norm, x)

    # DenseReluDense
    x = fwd_linear(ff_0, x)
    x = nn.relu(x)
    if dropout_key is not None:
        x = dropout(subkeys[1], x)
    x = fwd_linear(ff_1, x)
    if dropout_key is not None:
        x = dropout(subkeys[2], x)

    # Add
    x = x + _x

    return x
