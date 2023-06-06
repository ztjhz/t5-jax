import sys, pathlib

sys.path.append(str(pathlib.Path(__file__).parents[1]))

import jax
import jax.numpy as jnp
import jax.random as random

from model.t5 import fwd_t5
from utils.loss_utils import cross_entropy_loss
from utils.data_utils import dataset_generator
from utils.random_utils import key2seed
from utils.evaluation_utils import evaluate_model_with_sequential_generation
from config import config

import optax
import wandb


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
    mask_dec_1d = labels != config.PAD_TOKEN_ID
    loss = cross_entropy_loss(logits=logits, labels=labels, mask=mask_dec_1d)
    return loss


train_forward_and_backward = jax.value_and_grad(train_forward)


@jax.jit
def train_step(
    params: dict,
    encoder_input_ids: jnp.ndarray,
    decoder_input_ids: jnp.ndarray,
    labels: jnp.ndarray,
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
    params = optax.apply_updates(params, updates)

    return params, opt_state, loss


# eval step does not have dropout
@jax.jit
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
    mask_dec_1d = labels != config.PAD_TOKEN_ID
    loss = cross_entropy_loss(logits=logits, labels=labels, mask=mask_dec_1d)
    return loss


def main(params: dict):
    n_epochs = 1
    max_steps = 100_000
    eval_interval = 1024
    save_interval = 20480
    batch_size = 32
    lr = 1e-3

    wandb.init(
        project="t5-jax-fr-en-finetune",
        config={
            "learning_rate": lr,
            "batch size": batch_size,
            "optimizer": "adam",
            "dataset": "wmt14-train",
            "epochs": n_epochs,
            "max_steps": max_steps,
            "device": "tpu",
            "params": "init_params_embedding_lm_head",
        },
    )

    # set up optimizer
    global optimizer
    optimizer = optax.adamw(learning_rate=lr)
    opt_state = optimizer.init(params)

    train_generator = dataset_generator("train")
    eval_generator = dataset_generator("test")
    key = random.PRNGKey(2418)
    total_steps = 0

    for epoch in range(n_epochs):
        epoch_train_loss = 0
        total_train_steps = 0

        key, shuffle_key_train, shuffle_key_test = random.split(key, 3)

        train_set = train_generator.shuffle(seed=key2seed(shuffle_key_train))
        eval_set = eval_generator.shuffle(seed=key2seed(shuffle_key_test))

        for step, batch_train in enumerate(train_set.iter(batch_size=batch_size)):
            key, dropout_key = random.split(key)

            total_steps += 1
            if total_steps > max_steps:
                break

            params, opt_state, loss = train_step(
                params=params,
                encoder_input_ids=batch_train["encoder_input_ids"],
                decoder_input_ids=batch_train["decoder_input_ids"],
                labels=batch_train["labels"],
                opt_state=opt_state,
                dropout_key=dropout_key,
            )

            epoch_train_loss += loss
            total_train_steps += 1
            print(f"Epoch {epoch}, Step {step}, train loss {loss}")
            wandb.log({"train loss": loss})

            # eval
            if step % eval_interval == 0:
                total_eval_loss = 0
                total_eval_steps = 0
                for batch_eval in eval_set.iter(batch_size=batch_size):
                    loss = eval_step(
                        params=params,
                        encoder_input_ids=batch_eval["encoder_input_ids"],
                        decoder_input_ids=batch_eval["decoder_input_ids"],
                        labels=batch_eval["labels"],
                    )
                    total_eval_loss += loss
                    total_eval_steps += 1

                print(f"Epoch {epoch}, step {step}, Eval loss: {total_eval_loss / total_eval_steps}")
                wandb.log({"eval loss": total_eval_loss / total_eval_steps})

                # compute bleu score
                bleu = evaluate_model_with_sequential_generation(params)
                wandb.log({"bleu score": bleu})

            if step % save_interval == 0:
                jnp.save(f"{wandb.run.name}-{epoch}-{step}.npy", params)

        if total_steps > max_steps:
            break

        print(f"Epoch {epoch}, loss {epoch_train_loss / total_train_steps}")
        wandb.log({"epoch loss": epoch_train_loss / total_train_steps})


if __name__ == "__main__":
    from jax_smi import initialise_tracking
    from utils.params_utils import init_params_embedding_lm_head

    initialise_tracking()
    params = init_params_embedding_lm_head()
    main(params)
