from functools import partial
import jax
import jax.numpy as jnp
from jax import random

from model.transformer_encoder import fwd_transformer_encoder
from model.transformer_decoder import fwd_transformer_decoder
from model.linear import fwd_linear

from config import config


@partial(jax.jit, static_argnames="tie_word_embeddings")
def fwd_t5(
    params: dict,
    encoder_input_ids: jnp.ndarray,
    decoder_input_ids: jnp.ndarray,
    tie_word_embeddings: bool = True,
    encoder_output: jnp.ndarray = None,
    dropout_key: any = None,
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

        dropout_key (KeyArray, optional): A key to use for dropout. Default is `None`, which means no dropout is applied.

    Returns:
        logits (jnp.ndarray): The output logits of the T5 model. (batch_size, decoder_sequence_length, vocab_size)

        encoder_output (jnp.ndarray): The output of the encoder of the T5 model.
    """
    embedding_params = params["shared"]
    encoder_params = params["encoder"]
    decoder_params = params["decoder"]
    embeddings = embedding_params["embedding"]

    # Encoder self attention  mask
    mask_enc_1d = encoder_input_ids != config.PAD_TOKEN_ID
    mask_enc = jnp.einsum("bi,bj->bij", mask_enc_1d, mask_enc_1d)[:, None]

    # Decoder self attention mask (casual masking / auto regressive)
    n = decoder_input_ids.shape[1]
    lower_triangle = jnp.tri(n, dtype=jnp.bool_)
    upper_triangle = jnp.tri(n, k=1, dtype=jnp.bool_)
    mask_dec = jnp.reshape(
        jnp.einsum("ij,ij->ij", lower_triangle, upper_triangle), (1, 1, n, n)
    )

    # Encoder-decoder cross attention mask
    mask_dec_1d = jnp.ones(decoder_input_ids.shape, dtype=jnp.bool_)
    mask_enc_dec = jnp.einsum("bi,bj->bij", mask_dec_1d, mask_enc_1d)[:, None]

    # Set up dropout keys
    encoder_dropout_key, decoder_dropout_key = None, None
    if dropout_key is not None:
        encoder_dropout_key, decoder_dropout_key = random.split(dropout_key, 2)

    if encoder_output is None:
        encoder_output = fwd_transformer_encoder(
            encoder_params=encoder_params,
            embedding_params=embedding_params,
            input_ids=encoder_input_ids,
            dropout_key=encoder_dropout_key,
            mask=mask_enc,
        )

    decoder_output = fwd_transformer_decoder(
        decoder_params=decoder_params,
        embedding_params=embedding_params,
        decoder_input_ids=decoder_input_ids,
        encoder_output=encoder_output,
        dropout_key=decoder_dropout_key,
        self_attn_mask=mask_dec,
        cross_attn_mask=mask_enc_dec,
    )

    if tie_word_embeddings:
        # scale decoder output
        d_model = encoder_output.shape[-1]
        scaled_decoder_output = decoder_output / (d_model**0.5)

        logits = fwd_linear({"kernel": embeddings.T}, scaled_decoder_output)
    else:
        lm_head = params["lm_head"]
        logits = fwd_linear(params=lm_head, x=decoder_output)

    return logits, encoder_output
