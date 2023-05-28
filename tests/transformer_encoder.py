import jax.numpy as jnp

from transformers import AutoTokenizer, FlaxT5ForConditionalGeneration

from ..model.transformer_encoder import fwd_transformer_encoder
from ..model.layer_norm import fwd_layer_norm_rms
from ..model.embedding import fwd_embedding

tokenizer = AutoTokenizer.from_pretrained("t5-base")
model = FlaxT5ForConditionalGeneration.from_pretrained("allenai/unifiedqa-t5-base")

inputs = tokenizer(
    ["summarize: My friends are cool but they eat too many carbs."], return_tensors="np"
)
input_ids = inputs["input_ids"]

# flax output
output_flax = model.encode(
    input_ids, output_hidden_states=True, return_dict=True, output_attentions=True
)

# my output
mask_dec_1d = jnp.ones(input_ids.shape, dtype=jnp.bool_)
mask = jnp.einsum("bi,bj->bij", mask_dec_1d, mask_dec_1d)[:, None]

x = fwd_embedding(model.params["shared"], input_ids)

assert jnp.allclose(x, output_flax["hidden_states"][0]) == True

position_bias = None

for i in range(12):
    params = model.params["encoder"]["block"][str(i)]["layer"]
    x, position_bias = fwd_transformer_encoder(
        params, x, mask, position_bias=position_bias
    )
    print(i)
    assert jnp.allclose(x, output_flax["hidden_states"][i + 1]) == True


output = fwd_layer_norm_rms(model.params["encoder"]["final_layer_norm"], x)
assert jnp.allclose(output_flax["last_hidden_state"], output)
