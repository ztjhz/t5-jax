from typing import Dict

import jax.numpy as jnp


def fwd_relative_attention_bias(
    params: Dict,
    query_sequence_length: int,
    target_sequence_length: int,
    bidirectional: bool = True,
) -> jnp.ndarray:
    """
    Compute the forward relative attention bias (relative position representation) for transformer-based models.

    This implementation is based on the Self-Attention with Relative Position Representations paper:
    https://arxiv.org/abs/1803.02155.

    Args:
        params : Dict
            A dictionary containing the relative_attention_bias parameters

        query_sequence_length : int
            The length of the query sequence.

        target_sequence_length : int
            The length of the key sequence.

        bidirectional : bool, optional
            Whether the attention mechanism is bidirectional or not. Defaults to True.

    Returns:
        jnp.ndarray
            The relative attention bias for each head in the transformer model. The shape of the returned
            tensor is (1 (batch size), n_heads, query_sequence_length, target_sequence_length).
    """
    # shape: (relative_attention_num_buckets, n_heads)
    relative_attention_bias = params["embedding"]

    # (query_sequence_length, 1)
    context_position = jnp.arange(query_sequence_length, dtype="i4")[:, None]
    # (1, target_sequence_length)
    memory_position = jnp.arange(target_sequence_length, dtype="i4")[None, :]
    # (query_sequence_length, target_sequence_length)
    relative_position = memory_position - context_position

    relative_position_bucket = calculate_relative_position_bucket(
        relative_position, bidirectional
    )

    # (query_sequence_length, target_sequence_length, n_heads)
    bias = relative_attention_bias[relative_position_bucket]

    # (1 (batch size), n_heads, query_sequence_length, target_sequence_length)
    bias = bias.transpose((2, 0, 1))[None, :, :, :]

    return bias


def calculate_relative_position_bucket(
    relative_position: jnp.ndarray,
    bidirectional: bool = True,
    num_buckets=32,
    max_distance=128,
):
    """
    Args:
        relative_position (jnp.ndarray): The relative position between tokens, calculated as memory_position - query_position.
        bidirectional (bool, optional): Whether the attention mechanism is bidirectional. Default is True. (In decoder, where there is casual masking, bidirectional is false)
        num_buckets (int, optional): The number of buckets to use. Default is 32.
        max_distance (int, optional): The maximum distance to consider. Default is 128.

    Returns:
        relative_buckets (jnp.ndarray)

    Adapted from Mesh Tensorflow:
    https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

    Translate relative position to a bucket number for relative attention. The relative position is defined as
    memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
    position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
    small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
    positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
    This should allow for more graceful generalization to longer sequences than the model has been trained on
    """

    relative_buckets: jnp.ndarray = 0
    if bidirectional:
        # for bidirectional, buckets are divided into 2 categories: positive and negative relative positions
        num_buckets //= 2

        # preserve information about the original direction of relative_position
        relative_buckets += (relative_position > 0) * num_buckets

        # only keep the magnitude, not the direction
        relative_position = jnp.abs(relative_position)
    else:
        # non-birectional -> casual masking -> clip positive relative positions to 0, then convert negative to positive
        relative_position = -jnp.clip(relative_position, a_max=0)

    # relative_position is now in the range [0, inf)

    # half of the buckets are for exact increments in positions
    max_exact = num_buckets // 2

    # boolean mask for relative positions smaller than max_exact
    is_small = relative_position < max_exact

    # The other half of the buckets are for logarithmically bigger bins in positions from max_exact to max_distance
    # For larger relative positions, we compute their bucket number logarithmically
    # This is done to create larger buckets for larger distances, allowing for more graceful generalization to longer sequences
    relative_position_if_large = max_exact + (
        jnp.log(relative_position / max_exact)
        / jnp.log(max_distance / max_exact)
        * (num_buckets - max_exact)
    ).astype("i4")

    # ensure it doesn't exceed num_buckets
    relative_position_if_large = jnp.clip(
        relative_position_if_large, a_max=num_buckets - 1
    )

    # update relative_buckets with either the exact relative position or the logged relative position
    relative_buckets += jnp.where(
        is_small, relative_position, relative_position_if_large
    )

    return relative_buckets
