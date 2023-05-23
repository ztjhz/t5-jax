from typing import Dict

import jax.numpy as jnp


def split_projection_to_heads(params: Dict, n_head: int):
    """
    Splits the projection of a key into attention heads.

    Args:
        params (Dict): A dictionary containing the projection parameters.
        n_head (int): The number of attention heads.

    Returns:
        Dict: A dictionary with the split projection parameters, including reshaped kernel and bias if present.

    Raises:
        None

    Example:
        params = {
            "kernel": ndarray of shape (d_model, d_proj),
            "bias": ndarray of shape (d_model) (optional)
        }
        n_head = 12

        split_proj = split_projection_to_heads(params, n_head)
        Returns:
        {
            "kernel": ndarray of shape (d_model, n_head, d_proj // n_head),
            "bias": ndarray of shape (n_head, d_model // n_head) (optional, if bias is present)
        }
    """

    split_proj = {}
    # Reshape kernel for multi-head attention
    split_proj["kernel"] = jnp.reshape(
        params["kernel"],
        newshape=(
            params["kernel"].shape[0],
            n_head,
            params["kernel"].shape[1] // n_head,
        ),
    )

    # Reshape bias if present
    if "bias" in params:
        split_proj["bias"] = jnp.reshape(
            params["bias"],
            newshape=(
                n_head,
                params["bias"].shape[0] // n_head,
            ),
        )

    return split_proj
