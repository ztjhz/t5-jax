import sys, pathlib

sys.path.append(str(pathlib.Path(__file__).parents[1]))

import jax.numpy as jnp

from transformers import AutoTokenizer, FlaxT5ForConditionalGeneration

from model.transformer_decoder import fwd_transformer_decoder
from model.transformer_encoder import fwd_transformer_encoder


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
encoder_output_flax = model.encode(
    encoder_input_ids,
    return_dict=True,
)
decoder_output_flax = model.decode(
    decoder_input_ids,
    encoder_output_flax,
    output_hidden_states=True,
    return_dict=True,
)["hidden_states"][-1]

# my output
encoder_output = fwd_transformer_encoder(
    encoder_params=model.params["encoder"],
    embedding_params=model.params["shared"],
    input_ids=encoder_input_ids,
)
decoder_output = fwd_transformer_decoder(
    decoder_params=model.params["decoder"],
    embedding_params=model.params["shared"],
    decoder_input_ids=decoder_input_ids,
    encoder_output=encoder_output,
)

assert jnp.allclose(decoder_output_flax, decoder_output)
