import jax.numpy as jnp

from transformers import FlaxT5ForConditionalGeneration

from model.linear import init_linear
from config import config


def init_params_pretrained() -> jnp.ndarray:
    """
    Initializes the model parameters by loading a pretrained model from the "allenai/unifiedqa-t5-base" checkpoint.

    Returns:
        The model parameters.
    """
    model = FlaxT5ForConditionalGeneration.from_pretrained("allenai/unifiedqa-t5-base")
    return model.params


def init_params_random_lm_head() -> jnp.ndarray:
    """
    Initializes the model parameters with a randomly initialized linear language model head.

    Returns:
        The initialized model parameters with the randomly initialized linear head.
    """
    params = init_params_pretrained()
    params["lm_head"] = init_linear(shape=(config.D_MODEL, config.VOCAB_SIZE))
    return params


def init_params_embedding_lm_head() -> jnp.ndarray:
    """
    Initializes the model parameters with an embedding-based linear language model head.

    Returns:
        The initialized model parameters with the embedding-based linear head.
    """
    params = init_params_pretrained()
    params["lm_head"] = {"kernel": params["shared"]["embedding"].T}
    return params


def load_params(file_path: str) -> jnp.ndarray:
    """
    Loads the model parameters from a file.

    Args:
        file_path (str): The path to the file containing the model parameters.

    Returns:
        The loaded model parameters.
    """
    return jnp.load(file_path, allow_picky=True).item()
