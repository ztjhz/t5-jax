import jax.numpy as jnp

from transformers import AutoTokenizer
from datasets import DatasetDict, load_dataset

from config import config

tokenizer = AutoTokenizer.from_pretrained("t5-base")


def tokenize_text(text: list[str]) -> jnp.ndarray:
    return tokenizer(
        text,
        return_tensors="np",
        padding="max_length",
        truncation=True,
        max_length=128,
    ).input_ids


def transform_data(data: list[dict]) -> jnp.ndarray:
    en, fr = data["en"], data["fr"]
    encoder_input_ids, labels = (
        tokenize_text(fr),
        tokenize_text(en),
    )

    bos = (
        jnp.ones(shape=(labels.shape[0], 1), dtype=jnp.int32)
        * config.DECODER_START_TOKEN_ID
    )
    decoder_input_ids = jnp.concatenate([bos, labels], axis=-1)[..., :-1]

    return {
        "encoder_input_ids": encoder_input_ids,
        "decoder_input_ids": decoder_input_ids,
        "labels": labels,
    }


def process_dataset(dataset: DatasetDict) -> DatasetDict:
    return (
        dataset.flatten()
        .rename_columns({"translation.en": "en", "translation.fr": "fr"})
        .with_transform(transform_data)
    )


def dataset_generator(train: bool = True) -> DatasetDict:
    train_set = load_dataset("wmt14", "fr-en", split="validation" if train else "test")
    train_set = process_dataset(train_set)
    return train_set
