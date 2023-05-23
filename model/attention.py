from typing import Dict, List

import jax.nn as jnn
import jax.numpy as jnp

from ..utils.attention_utils import split_projection_to_heads


# model.params['encoder']['block']['1']['layer']['0']['SelfAttention']['q']['kernel']
# model.params['encoder']['block']['0']['layer']['0']['SelfAttention']['relative_attention_bias']['embedding'].shape
# layer 0 has relative_attention_bias https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_flax_t5.py#L225
def fwd_attention(
    params: Dict, qry_states: List[float], tgt_states: List[float], mask: List[bool]
) -> List[float]:
    """
    Perform forward attention computation.

    Args:
        params (Dict): A dictionary containing additional parameters for attention computation.

        qry_states (List[float]): The query tensor with shape (batch_size, query_sequence_length, d_model).

        tgt_states (List[float]): The key and value tensor with shape (batch_size, target_sequence_length, d_model).

        mask (List[bool]): The attention mask tensor with shape (batch_size, 1, query_sequence_length, target_sequence_length).

    Returns:
        List[float]: The computed attention scores as a list of floats.
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

    if "bias" in q_proj:
        q += q_proj["bias"]
        k += k_proj["bias"]
        v += v_proj["bias"]

    # QK matrix multiplication
    # (batch_size, n_heads, query_sequence_length, target_sequence_length)
    qk = jnp.einsum("bqhk,bthk->bhqt", q, k)
    # Scale
    qk = qk / jnp.sqrt(d_k)
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

    if "bias" in out:
        output += out["bias"]

    return output
