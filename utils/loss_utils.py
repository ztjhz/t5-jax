import jax.numpy as jnp
import optax


def cross_entropy_loss(
    logits: jnp.ndarray, labels: jnp.ndarray, mask: jnp.ndarray
) -> jnp.ndarray:
    """
    Computes the cross-entropy loss for a given set of predictions (logits) and true labels.

    Args:
        logits (jnp.ndarray): The predicted class logits from the model.
            It is a 3D array of shape (batch_size, sequence_length, n_classes), where 'n_classes' is the number of possible classes for each item in the sequence.
            Each innermost array contains the logits (unnormalized log probabilities) for the corresponding class.

        labels (jnp.ndarray): The true class labels.
            It is a 2D array of shape (batch_size, sequence_length), where each entry is the integer label of the correct class for the corresponding sequence item.

        mask (jnp.ndarray): A mask array of shape (batch_size, sequence_length), used to ignore certain elements in the loss computation. This could be useful for sequences of differing lengths in the batch, where padding has been used.

    Returns:
        jnp.ndarray: A 0 dimension array containing the computed cross-entropy loss (a single floating-point value)
    """
    # loss shape: (batch_size, sequence_length)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=labels)
    return jnp.mean(loss, where=mask)
