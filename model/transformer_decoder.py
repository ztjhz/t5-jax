from typing import Dict, List

import jax.numpy as jnp
from jax import random
import jax.nn as nn

from ..model.attention import fwd_attention
from ..model.layer_norm import fwd_layer_norm
from ..model.dropout import dropout
from ..model.linear import fwd_linear


def fwd_transformer_decoder(
    params: Dict,
    qry_states: List[float],
    tgt_states: List[float],
    mask: List[bool],
    dropout_key: List = None,
):
    self_attn_block, cross_attn_block, ff_block = params["0"], params["1"], params["2"]

    self_attn = self_attn_block["SelfAttention"]
    self_attn_layer_norm = self_attn_block["layer_norm"]

    cross_attn = cross_attn_block["EncDecAttention"]
    cross_attn_layer_norm = cross_attn_block["layer_norm"]

    ff_0 = ff_block["DenseReluDense"]["wi"]
    ff_1 = ff_block["DenseReluDense"]["wo"]
    final_layer_norm = ff_block["layer_norm"]

    if dropout_key is not None:
        subkeys = random.split(dropout_key, 4)

    # Multi-head self attention
    _qry_states = qry_states
    qry_states = fwd_attention(self_attn, qry_states, qry_states, mask)

    # Dropout
    if dropout_key is not None:
        qry_states = dropout(subkeys[0], qry_states)

    # Add & Norm
    qry_states += _qry_states
    qry_states = fwd_layer_norm(self_attn_layer_norm, qry_states)

    # Multi-head cross attention
    _qry_states = qry_states
    x = fwd_attention(cross_attn, qry_states, tgt_states, mask)

    # Dropout
    if dropout_key is not None:
        x = dropout(subkeys[1], x)

    # Add & Norm
    x = x + _qry_states
    x = fwd_layer_norm(cross_attn_layer_norm, x)

    # Feed Forward (Linear -> Gelu -> Dropout -> Linear -> Dropout)
    _x = x
    x = fwd_linear(ff_0, x)
    x = nn.gelu(x)
    if dropout_key is not None:
        x = dropout(subkeys[2], x)
    x = fwd_linear(ff_1, x)
    if dropout_key is not None:
        x = dropout(subkeys[3], x)

    # Add & Norm
    x = x + _x
    x = fwd_layer_norm(final_layer_norm, x)

    return x
