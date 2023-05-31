import jax.numpy as jnp

from model.transformer_encoder_block import fwd_transformer_encoder_block
from model.layer_norm import fwd_layer_norm_rms
from model.embedding import fwd_embedding


def fwd_transformer_encoder(
    encoder_params: dict, embedding_params: dict, input_ids: jnp.ndarray
):
    # create mask
    mask_dec_1d = jnp.ones(input_ids.shape, dtype=jnp.bool_)
    mask = jnp.einsum("bi,bj->bij", mask_dec_1d, mask_dec_1d)[:, None]

    # convert inputs to embeddings
    x = fwd_embedding(embedding_params, input_ids)

    position_bias = None

    # 12 layers in the encoder
    for i in range(12):
        params = encoder_params["block"][str(i)]["layer"]
        x, position_bias = fwd_transformer_encoder_block(
            params, x, mask=mask, position_bias=position_bias
        )

    # final layer norm
    output = fwd_layer_norm_rms(encoder_params["final_layer_norm"], x)

    return output
