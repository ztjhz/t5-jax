import jax.numpy as jnp
import jax.nn as jnn

from model.transformer_encoder import fwd_transformer_encoder
from model.transformer_decoder import fwd_transformer_decoder
from model.linear import fwd_linear


def fwd_t5(
    params: dict,
    encoder_input_ids: jnp.ndarray,
    decoder_input_ids: jnp.ndarray,
    tie_word_embeddings: bool = True,
    encoder_output: jnp.ndarray = None,
):
    """
    Run forward propagation through the T5 model.

    Args:
        params (dict): A dictionary containing the parameters of the T5 model. It must contain
                       "shared", "encoder", "decoder" (and optionally "lm_head" if `tie_word_embeddings`
                       is False) as keys.

        encoder_input_ids (jnp.ndarray): The input ids to be passed to the encoder.

        decoder_input_ids (jnp.ndarray): The input ids to be passed to the decoder.

        tie_word_embeddings (bool, optional): Whether to use tie word embeddings for the language model head.
                                              It scales the decoder output and uses the word embeddings for computing the logits.
                                              True when training and false when fine tuning. Defaults to True.

        encoder_output (jnp.ndarray, optional): If provided, this output is used instead of re-computing the encoder output. Default is None.
                                                This is to cache the encoder output from previous runs

    Returns:
        logits (jnp.ndarray): The output logits of the T5 model.
    """
    embedding_params = params["shared"]
    encoder_params = params["encoder"]
    decoder_params = params["decoder"]
    embeddings = embedding_params["embedding"]

    if encoder_output is None:
        encoder_output = fwd_transformer_encoder(
            encoder_params=encoder_params,
            embedding_params=embedding_params,
            input_ids=encoder_input_ids,
        )

    decoder_output = fwd_transformer_decoder(
        decoder_params=decoder_params,
        embedding_params=embedding_params,
        decoder_input_ids=decoder_input_ids,
        encoder_output=encoder_output,
    )

    if tie_word_embeddings:
        # scale decoder output
        d_model = encoder_output.shape[-1]
        scaled_decoder_output = decoder_output / (d_model**0.5)

        logits = fwd_linear({"kernel": embeddings.T}, scaled_decoder_output)
    else:
        lm_head = params["lm_head"]
        logits = fwd_linear(params=lm_head, x=scaled_decoder_output)

    return logits
