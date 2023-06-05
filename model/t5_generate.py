import jax.numpy as jnp

from model.t5 import fwd_t5
from config import config


def fwd_t5_generate(
    params: dict,
    encoder_input_ids: jnp.ndarray,
    eos_token_id: int = 1,
    decoder_start_token_id: int = 0,
    tie_word_embeddings: bool = True,
) -> jnp.ndarray:
    """
    Generates a sequence using the fwd_t5.

    Args:
        params (dict): A dictionary of model parameters.

        encoder_input_ids (jnp.ndarray): Input IDs for the encoder.

        eos_token_id (int, optional): ID of the end-of-sequence token. Defaults to 1.

        decoder_start_token_id (int, optional): ID of the start-of-sequence token for the decoder. Defaults to 0.

        tie_word_embeddings (bool, optional): Whether to use tie word embeddings for the language model head. Defaults to True.

    Returns:
        jnp.ndarray: Sequence generated by the T5 model. (batch_size, sequence_length)

    """
    token = None
    encoder_output = None
    # shape: (batch_size, decoder_sequence_length)
    decoder_input_ids = (
        jnp.ones((encoder_input_ids.shape[0], 1), dtype="i4") * decoder_start_token_id
    )
    # to keep track of whether a sequence has reached EOS (batch_size, 1)
    is_complete = jnp.zeros((encoder_input_ids.shape[0], 1), dtype=jnp.bool_)

    i = 0
    while i < config.MAX_GENERATION_LENGTH and not jnp.all(is_complete):
        i += 1
        # logits shape: (batch_size, decoder_sequence_length, vocab_size)
        logits, encoder_output = fwd_t5(
            params=params,
            encoder_input_ids=encoder_input_ids,
            decoder_input_ids=decoder_input_ids,
            tie_word_embeddings=tie_word_embeddings,
            encoder_output=encoder_output,
        )
        # compute the token for all logits of the decoder sequence output
        # token shape: (batch_size, decoder_sequence_length)
        token = jnp.argmax(logits, axis=-1)

        # only keep the last token of each decoder sequence output
        # token shape: (batch_size,)
        token = token[..., -1]

        # reshape to (batch_size, 1)
        token = token[:, None]

        # convert token of completed sequence to pad
        token = token * ~is_complete + config.PAD_TOKEN_ID * is_complete

        # update whether a sequence has reached EOS
        is_complete = is_complete | token == eos_token_id

        # add current token output to the decoder_input_ids
        decoder_input_ids = jnp.concatenate([decoder_input_ids, token], axis=-1)

    return decoder_input_ids
