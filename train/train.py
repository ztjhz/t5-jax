import sys, pathlib

sys.path.append(str(pathlib.Path(__file__).parents[1]))

import jax
import jax.numpy as jnp
import jax.random as random

from model.t5 import fwd_t5
from utils.loss_utils import cross_entropy_loss
from utils.data_utils import dataset_generator

import optax


def train_forward(
    params: dict,
    encoder_input_ids: jnp.ndarray,
    decoder_input_ids: jnp.ndarray,
    labels: jnp.ndarray,
    dropout_key: any,
) -> jnp.ndarray:
    logits, _ = fwd_t5(
        params=params,
        encoder_input_ids=encoder_input_ids,
        decoder_input_ids=decoder_input_ids,
        tie_word_embeddings=False,
        dropout_key=dropout_key,
    )
    mask_dec_1d = jnp.ones(decoder_input_ids.shape, dtype=jnp.bool_)
    loss = cross_entropy_loss(logits=logits, labels=labels, mask=mask_dec_1d)
    return loss


train_forward_and_backward = jax.value_and_grad(train_forward)


def train_step(
    params: dict,
    encoder_input_ids: jnp.ndarray,
    decoder_input_ids: jnp.ndarray,
    labels: jnp.ndarray,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
    dropout_key: any,
) -> tuple[dict, optax.OptState, jnp.ndarray]:
    loss, grads = train_forward_and_backward(
        params,
        encoder_input_ids=encoder_input_ids,
        decoder_input_ids=decoder_input_ids,
        labels=labels,
        dropout_key=dropout_key,
    )

    updates, opt_state = optimizer.update(grads, opt_state, params)
    optax.apply_updates(params, updates)

    return params, opt_state, loss


# eval step does not have dropout
def eval_step(
    params: dict,
    encoder_input_ids: jnp.ndarray,
    decoder_input_ids: jnp.ndarray,
    labels: jnp.ndarray,
) -> jnp.ndarray:
    logits, _ = fwd_t5(
        params=params,
        encoder_input_ids=encoder_input_ids,
        decoder_input_ids=decoder_input_ids,
        tie_word_embeddings=False,
    )
    mask_dec_1d = jnp.ones(decoder_input_ids.shape, dtype=jnp.bool_)
    loss = cross_entropy_loss(logits=logits, labels=labels, mask=mask_dec_1d)
    return loss


def main(params: dict):
    # set up optimizer
    optimizer = optax.adafactor(learning_rate=0.001)
    opt_state = optimizer.init(params)

    n_epochs = 1
    eval_interval = 1024
    batch_size = 1

    train_generator = dataset_generator(train=True)
    eval_generator = dataset_generator(train=False)
    key = random.PRNGKey(2418)

    for epoch in range(n_epochs):
        epoch_train_loss = 0

        train_set, eval_set = train_generator.shuffle(), eval_generator.shuffle()

        for step, batch_train in enumerate(train_set.iter(batch_size=batch_size)):
            key, dropout_key = random.split(key)

            params, opt_state, loss = train_step(
                params=params,
                encoder_input_ids=batch_train["encoder_input_ids"],
                decoder_input_ids=batch_train["decoder_input_ids"],
                labels=batch_train["labels"],
                optimizer=optimizer,
                opt_state=opt_state,
                dropout_key=dropout_key,
            )

            epoch_train_loss += loss
            print(f"Step {step}, loss {loss}")

            # eval
            if step % eval_interval == 0:
                total_loss = 0
                for batch_eval in eval_set.iter(batch_size=batch_size):
                    loss = eval_step(
                        params=params,
                        encoder_input_ids=batch_eval["encoder_input_ids"],
                        decoder_input_ids=batch_eval["decoder_input_ids"],
                        labels=batch_eval["labels"],
                    )
                    total_loss += loss

                print(f"Epoch {epoch}, step {step}, Total loss: {total_loss}")

        print(f"Epoch {epoch}, loss {epoch_train_loss}")
