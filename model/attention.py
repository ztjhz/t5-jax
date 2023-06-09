from typing import Dict, List, Tuple

import jax.nn as jnn
import jax.numpy as jnp

from utils.attention_utils import split_projection_to_heads


def fwd_attention(
    params: Dict,
    qry_states: List[float],
    tgt_states: List[float],
    mask: List[bool],
    position_bias: jnp.ndarray = None,
    scale: bool = False,
) -> jnp.ndarray:
    """
    Perform forward attention computation.

    Args:
        params (Dict): A dictionary containing additional parameters for attention computation.

        qry_states (List[float]): The query tensor with shape (batch_size, query_sequence_length, d_model).

        tgt_states (List[float]): The key and value tensor with shape (batch_size, target_sequence_length, d_model).

        mask (List[bool]): The attention mask tensor with shape (batch_size, 1, query_sequence_length, target_sequence_length).

        position_bias (jnp.ndarray[Float], optional): The relative position bias (relative position embeddings) tensor with shape (batch size, n_heads, query_sequence_length, target_sequence_length). If none, no bias is added.

        scale (bool, optional). Whether to scale the qk matrix  by d_k. Defaults to False.

    Returns:
        attention_output (jnp.ndarray[Float]): The computed attention scores as a tensor of floats with shape (batch_size, query_sequence_length, d_out)
    """
    n_head = 12
    out = params["o"]

    # Multi-head attention
    # Reshape kernel and bias of q_proj, k_proj, v_proj for multi-head attention
    # Kernel: (d_model, d_proj) -> (d_model, n_head, d_proj // n_head {d_k or d_v})
    # Bias: (d_proj) -> (n_head,  d_proj // n_head {d_k or d_v})
    q_proj = split_projection_to_heads(params["q"], n_head)
    k_proj = split_projection_to_heads(params["k"], n_head)
    v_proj = split_projection_to_heads(params["v"], n_head)

    d_k = q_proj["kernel"].shape[-1]

    # 1. Linear layer
    # (batch_size, query_sequence_length, n_heads, d_k)
    q = jnp.einsum("bqm,mhk->bqhk", qry_states, q_proj["kernel"])
    # (batch_size, target_sequence_length, n_heads, d_k)
    k = jnp.einsum("btm,mhk->bthk", tgt_states, k_proj["kernel"])
    # (batch_size, target_sequence_length, n_heads, d_v)
    v = jnp.einsum("btm,mhv->bthv", tgt_states, v_proj["kernel"])

    # QK matrix multiplication
    # (batch_size, n_heads, query_sequence_length, target_sequence_length)
    qk = jnp.einsum("bqhk,bthk->bhqt", q, k)
    # Scale (scaling not done)
    if scale:
        qk = qk / jnp.sqrt(d_k)
    # Relative position attention bias
    if position_bias is not None:
        qk += position_bias
    # Mask
    qk = jnp.where(mask, qk, jnp.NINF)
    # SoftMax
    qk = jnn.softmax(qk)
    qk = jnp.where(mask, qk, 0)

    # QKV matrix multiplication
    # (n_heads, d_k, batch_size, query_sequence_length)
    qkv = jnp.einsum("bhqt,bthv->hvbq", qk, v)

    # hvbq -> (h*v)bq
    # (concatenated_heads, batch_size, query_sequence_length)
    qkv_concat = jnp.concatenate(qkv, 0)

    # (batch_size, query_sequence_length, d_out)
    output = jnp.einsum("cbq,cm->bqm", qkv_concat, out["kernel"])

    return output
