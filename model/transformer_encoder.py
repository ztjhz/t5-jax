import jax.numpy as jnp
import jax.random as random

from model.transformer_encoder_block import fwd_transformer_encoder_block
from model.layer_norm import fwd_layer_norm_rms
from model.embedding import fwd_embedding
from model.dropout import dropout


def fwd_transformer_encoder(
    encoder_params: dict,
    embedding_params: dict,
    input_ids: jnp.ndarray,
    mask: jnp.ndarray,
    dropout_key: list = None,
):
    # convert inputs to embeddings
    x = fwd_embedding(embedding_params, input_ids)

    # Dropout
    key = dropout_key
    if key is not None:
        dropout_key, key = random.split(key)
        x = dropout(dropout_key, x)

    position_bias = None

    # 12 layers in the encoder
    for i in range(12):
        if key is not None:
            dropout_key, key = random.split(key)

        params = encoder_params["block"][str(i)]["layer"]
        x, position_bias = fwd_transformer_encoder_block(
            params, x, mask=mask, position_bias=position_bias, dropout_key=dropout_key
        )

    # final layer norm
    output = fwd_layer_norm_rms(encoder_params["final_layer_norm"], x)

    # Dropout
    if key is not None:
        dropout_key, key = random.split(key)
        output = dropout(dropout_key, output)

    return output
