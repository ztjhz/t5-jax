import jax.numpy as jnp
import jax.random as random

from model.embedding import fwd_embedding
from model.transformer_decoder_block import fwd_transformer_decoder_block
from model.layer_norm import fwd_layer_norm_rms
from model.dropout import dropout


def fwd_transformer_decoder(
    decoder_params: dict,
    embedding_params: dict,
    decoder_input_ids: jnp.ndarray,
    encoder_output: jnp.ndarray,
    self_attn_mask: jnp.ndarray,
    cross_attn_mask: jnp.ndarray,
    dropout_key: list = None,
):
    # Convert inputs to embeddings
    x = fwd_embedding(embedding_params, decoder_input_ids)

    # Dropout
    key = dropout_key
    if key is not None:
        dropout_key, key = random.split(key)
        x = dropout(dropout_key, x)

    position_bias = None

    # 12 layers in the decoder
    for i in range(12):
        if key is not None:
            dropout_key, key = random.split(key)

        params = decoder_params["block"][str(i)]["layer"]
        x, position_bias = fwd_transformer_decoder_block(
            params=params,
            qry_states=x,
            tgt_states=encoder_output,
            self_attn_mask=self_attn_mask,
            cross_attn_mask=cross_attn_mask,
            position_bias=position_bias,
            dropout_key=dropout_key,
        )

    # Final layer norm
    output = fwd_layer_norm_rms(decoder_params["final_layer_norm"], x)

    # Dropout
    if key is not None:
        dropout_key, key = random.split(key)
        output = dropout(dropout_key, output)

    return output
