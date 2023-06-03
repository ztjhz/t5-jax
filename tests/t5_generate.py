import time
import sys, pathlib

sys.path.append(str(pathlib.Path(__file__).parents[1]))

from transformers import AutoTokenizer, FlaxT5ForConditionalGeneration

from model.t5_generate import fwd_t5_generate


tokenizer = AutoTokenizer.from_pretrained("t5-base")
model = FlaxT5ForConditionalGeneration.from_pretrained("allenai/unifiedqa-t5-base")

inputs = tokenizer(
    [
        "translate English to German: That is good.",
        "cola sentence: The course is jumping well.",
        "stsb sentence1: The rhino grazed on the grass. sentence2: A rhino is grazing in a field.",
        "summarize: In recent times, rapid advancements in technology have revolutionized various industries, enhancing efficiency, connectivity, and convenience for individuals and businesses alike.",
    ],
    return_tensors="np",
    padding=True,
)
input_ids, attention_mask = inputs.input_ids, inputs.attention_mask

# flax output
start_time_flax = time.time()
sequences_flax = None
for i in range(100):
    sequences_flax = model.generate(input_ids, attention_mask=attention_mask)["sequences"]
end_time_flax = time.time()
output_flax = tokenizer.batch_decode(sequences_flax, skip_special_tokens=True)

# my output
start_time = time.time()
sequences = None
for i in range(100):
    sequences = fwd_t5_generate(
        model.params,
        encoder_input_ids=input_ids,
        eos_token_id=model.config.eos_token_id,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )
end_time = time.time()
output = tokenizer.batch_decode(sequences, skip_special_tokens=True)

# check the outputs
print("Hugging Face output")
print(output_flax)
print("My output")
print(output)

# time taken
time_taken_flax = end_time_flax - start_time_flax
time_taken = end_time - start_time
diff = (time_taken_flax - time_taken) / time_taken_flax * 100

print("Time taken")
print(
    f"Hugging Face: {time_taken_flax:.2f}, Mine: {time_taken:.2f} ({diff:.2f}% faster)"
)
