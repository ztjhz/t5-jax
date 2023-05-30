import jax.numpy as jnp

from transformers import AutoTokenizer, FlaxT5ForConditionalGeneration
from model.embedding import fwd_embedding
from model.transformer_decoder import fwd_transformer_decoder
from model.layer_norm import fwd_layer_norm_rms

from tests.transformer_encoder import (
    output_flax as encoder_output_flax,
    output as encoder_output,
)

tokenizer = AutoTokenizer.from_pretrained("t5-base")
model = FlaxT5ForConditionalGeneration.from_pretrained("allenai/unifiedqa-t5-base")

inputs = tokenizer(
    ["summarize: My friends are cool but they eat too many carbs."], return_tensors="np"
)

encoder_input_ids = inputs.input_ids

decoder_start_token_id = model.config.decoder_start_token_id
decoder_input_ids = (
    jnp.ones((encoder_input_ids.shape[0], 1), dtype="i4") * decoder_start_token_id
)

# flax output
output_flax = model.decode(
    decoder_input_ids,
    encoder_output_flax,
    output_hidden_states=True,
    return_dict=True,
)

# my output
# self attention mask (casual masking)
n = decoder_input_ids.shape[1]
lower_triangle = jnp.tri(n, dtype=jnp.bool_)
upper_triangle = jnp.tri(n, k=1, dtype=jnp.bool_)
self_attn_mask = jnp.reshape(
    jnp.einsum("ij,ij->ij", lower_triangle, upper_triangle), (1, 1, n, n)
)

# cross attention mask
mask_enc_1d = jnp.ones(encoder_input_ids.shape, dtype=jnp.bool_)
mask_dec_1d = jnp.ones(decoder_input_ids.shape, dtype=jnp.bool_)
cross_attn_mask = jnp.einsum("bi,bj->bij", mask_dec_1d, mask_enc_1d)[:, None]

x = fwd_embedding(model.params["shared"], decoder_input_ids)

assert jnp.allclose(x, output_flax["hidden_states"][0]) == True

position_bias = None

for i in range(12):
    params = model.params["decoder"]["block"][str(i)]["layer"]
    x, position_bias = fwd_transformer_decoder(
        params=params,
        qry_states=x,
        tgt_states=encoder_output,
        self_attn_mask=self_attn_mask,
        cross_attn_mask=cross_attn_mask,
        position_bias=position_bias,
    )
    print(i)
    assert jnp.allclose(x, output_flax["hidden_states"][i + 1]) == True

output = fwd_layer_norm_rms(model.params["decoder"]["final_layer_norm"], x)
assert jnp.allclose(output_flax["hidden_states"][12], output)
