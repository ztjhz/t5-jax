import jax.numpy as jnp

from transformers import AutoTokenizer, FlaxT5ForConditionalGeneration
from model.embedding import fwd_embedding


tokenizer = AutoTokenizer.from_pretrained("t5-base")
model = FlaxT5ForConditionalGeneration.from_pretrained("allenai/unifiedqa-t5-base")

inputs = tokenizer(["summarize: My friends are cool but they eat too many carbs."], return_tensors="np")
input_ids = inputs["input_ids"]

embedding_layer = model.encode(input_ids, return_dict=True, output_hidden_states=True)["hidden_states"][0]
res_pretrained = embedding_layer[0]

embedding_pretrained = model.params["shared"]["embedding"]

params = {"embedding": embedding_pretrained}
res = fwd_embedding(params, input_ids)

equal = jnp.allclose(res, res_pretrained)
print(f"fwd_embedding: {equal}")

assert equal == True, "fwd_embedding is incorrect!"