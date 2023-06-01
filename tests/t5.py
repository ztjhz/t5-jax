import jax.numpy as jnp

from transformers import AutoTokenizer, FlaxT5ForConditionalGeneration

from model.t5 import fwd_t5


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
encoder_output_flax = model.encode(encoder_input_ids)
decoder_output_flax = model.decode(decoder_input_ids, encoder_output_flax)
logits_flax = decoder_output_flax["logits"]
sequences_flax = [jnp.argmax(logits_flax)]

output_flax = tokenizer.batch_decode(sequences_flax, skip_special_tokens=True)

# my output
logits, _ = fwd_t5(model.params, encoder_input_ids, decoder_input_ids)
sequences = [jnp.argmax(logits)]
output = tokenizer.batch_decode(sequences, skip_special_tokens=True)

# check the outputs
output_flax, output
