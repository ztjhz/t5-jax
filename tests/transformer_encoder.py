import jax

jax.config.update("jax_platforms", "cpu")

import sys, pathlib

sys.path.append(str(pathlib.Path(__file__).parents[1]))

import jax.numpy as jnp

from transformers import AutoTokenizer, FlaxT5ForConditionalGeneration

from model.transformer_encoder import fwd_transformer_encoder
from config import config


tokenizer = AutoTokenizer.from_pretrained("t5-base")
model = FlaxT5ForConditionalGeneration.from_pretrained("allenai/unifiedqa-t5-base")

inputs = tokenizer(
    ["summarize: My friends are cool but they eat too many carbs."], return_tensors="np"
)
input_ids = inputs["input_ids"]

# flax output
output_flax = model.encode(
    input_ids, output_hidden_states=True, return_dict=True, output_attentions=True
)["last_hidden_state"]

# my output
mask_1d = input_ids != config.PAD_TOKEN_ID
mask = jnp.einsum("bi,bj->bij", mask_1d, mask_1d)[:, None]

output = fwd_transformer_encoder(
    encoder_params=model.params["encoder"],
    embedding_params=model.params["shared"],
    input_ids=input_ids,
    mask=mask,
)

assert jnp.allclose(output_flax, output)
