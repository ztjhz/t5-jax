import sys, pathlib

sys.path.append(str(pathlib.Path(__file__).parents[1]))

import jax.numpy as jnp
from transformers import AutoTokenizer

from utils.metrics_utils import calculate_bleu_score
from utils.data_utils import dataset_generator

from model.t5_generate import fwd_t5_generate
from model.t5 import fwd_t5

from config import config


def evaluate_model_with_complete_decoding(params: jnp.ndarray) -> float:
    """
    (For fast evaluation)
    Evaluates the model using BLEU score on the wmt14 fr-en validation split.

    Performs only 1 pass through the T5 model by feeding it the right shifted decoder input ids.

    Args:
        params (jnp.ndarray): The model parameters.

    Returns:
        float: The BLEU score of the model.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        "t5-base", model_max_length=config.MAX_GENERATION_LENGTH
    )
    all_sequences = None
    all_encoded_labels = None
    validation_set = dataset_generator("validation")

    for i, data in enumerate(validation_set.iter(batch_size=32)):
        print(f"Step {i}")
        logits, _ = fwd_t5(
            params=params,
            encoder_input_ids=data["encoder_input_ids"],
            decoder_input_ids=data["decoder_input_ids"],
            tie_word_embeddings=False,
        )
        labels = data["labels"]
        mask = labels != config.PAD_TOKEN_ID
        token = jnp.argmax(logits, axis=-1)
        token = jnp.where(mask, token, config.PAD_TOKEN_ID)

        if all_sequences is None and all_encoded_labels is None:
            all_sequences = token
            all_encoded_labels = data["labels"]
        else:
            all_sequences = jnp.concatenate([all_sequences, token])
            all_encoded_labels = jnp.concatenate([all_encoded_labels, data["labels"]])

    predictions = tokenizer.batch_decode(all_sequences, skip_special_tokens=True)
    references = tokenizer.batch_decode(all_encoded_labels, skip_special_tokens=True)

    return calculate_bleu_score(predictions=predictions, references=references)


def evaluate_model_with_sequential_generation(params: jnp.ndarray) -> float:
    """
    Evaluates the model using BLEU score on the wmt14 fr-en validation split.

    This function performs generation with the T5 model starting with the <BOS> token.

    Args:
        params (jnp.ndarray): The model parameters.

    Returns:
        float: The BLEU score of the model.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        "t5-base", model_max_length=config.MAX_GENERATION_LENGTH
    )
    all_sequences = None
    all_encoded_labels = None
    validation_set = dataset_generator("validation")

    for i, data in enumerate(validation_set.iter(batch_size=32)):
        print(f"Step {i}")
        sequences = fwd_t5_generate(
            params, data["encoder_input_ids"], tie_word_embeddings=False
        )

        if all_sequences is None and all_encoded_labels is None:
            all_sequences = sequences
            all_encoded_labels = data["labels"]
        else:
            all_sequences = jnp.concatenate([all_sequences, sequences])
            all_encoded_labels = jnp.concatenate([all_encoded_labels, data["labels"]])

    predictions = tokenizer.batch_decode(all_sequences, skip_special_tokens=True)
    references = tokenizer.batch_decode(all_encoded_labels, skip_special_tokens=True)

    return calculate_bleu_score(predictions=predictions, references=references)
