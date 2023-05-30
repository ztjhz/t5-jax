from typing import Dict, List, Tuple

import jax.nn as jnn
import jax.numpy as jnp
from jax import random

from model.attention import fwd_attention
from model.dropout import dropout
from model.layer_norm import fwd_layer_norm_rms
from model.linear import fwd_linear
from model.relative_attention_bias import fwd_relative_attention_bias


def fwd_transformer_encoder(
    params: Dict,
    qry_states: jnp.ndarray,
    mask: jnp.ndarray,
    dropout_key: List = None,
    position_bias: jnp.ndarray = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Function implementing a forward pass through a Transformer encoder.

    Args:
        params (Dict): Parameters dictionary for transformer encoder

        qry_states (jnp.ndarray[float]): Query states to be passed to the self-attention module.

        mask (jnp.ndarray[bool]): Boolean mask for input sequence. `True` for valid positions, and `False` for positions to be masked.

        dropout_key (List, optional): A key to use for dropout. Default is `None`, which means no dropout is applied.

        position_bias (jnp.ndarray[float], optional): Precomputed position bias for relative positional encoding. If `None` (1st layer), it will be calculated inside this function.

    Returns:
        output_states (jnp.ndarray[float])

        position_bias (jnp.ndarray[float])
    """
    self_attn_block, ff_block = params["0"], params["1"]

    self_attn = self_attn_block["SelfAttention"]
    self_attn_layer_norm = self_attn_block["layer_norm"]

    ff_0 = ff_block["DenseReluDense"]["wi"]
    ff_1 = ff_block["DenseReluDense"]["wo"]
    ff_layer_norm = ff_block["layer_norm"]

    if dropout_key is not None:
        subkeys = random.split(dropout_key, 3)

    # Pre layer norm
    normed_qry_states = fwd_layer_norm_rms(self_attn_layer_norm, qry_states)

    # relative attention bias (relative position representation)
    # only compute for the first layer
    if position_bias is None:
        query_sequence_length = normed_qry_states.shape[1]
        target_sequence_length = query_sequence_length
        # (batch_size, n_head, query_sequence_length, target_sequence_length)
        position_bias = fwd_relative_attention_bias(
            self_attn["relative_attention_bias"],
            query_sequence_length,
            target_sequence_length,
        )

    # Multi-head attention with relative attention bias (relative position representations)
    x = fwd_attention(
        self_attn, normed_qry_states, normed_qry_states, mask, position_bias
    )

    # Dropout
    if dropout_key is not None:
        x = dropout(subkeys[0], x)

    # Add
    x = x + qry_states

    # Feed Forward (Linear -> Relu -> Dropout -> Linear -> Dropout)
    _x = x

    # Pre layer norm
    x = fwd_layer_norm_rms(ff_layer_norm, x)

    # DenseReluDense
    x = fwd_linear(ff_0, x)
    x = jnn.relu(x)
    if dropout_key is not None:
        x = dropout(subkeys[1], x)
    x = fwd_linear(ff_1, x)
    if dropout_key is not None:
        x = dropout(subkeys[2], x)

    # Add
    x = x + _x

    return x, position_bias
