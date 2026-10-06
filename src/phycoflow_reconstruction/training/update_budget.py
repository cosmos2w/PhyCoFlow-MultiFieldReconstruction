"""Keep reporting epochs separate from the uninterrupted shuffled data stream."""

import math


def post_training_final_step(configured_steps, start_step, steps_per_epoch, *,
                             max_steps=None, until_epoch=None, additional_epochs=None):
    """Resolve a bounded invocation without changing the saved training recipe."""
    limits = (max_steps, until_epoch, additional_epochs)
    if sum(value is not None for value in limits) > 1:
        raise ValueError("choose only one of max_steps, until_epoch, additional_epochs")
    for value in limits:
        if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
            raise ValueError("post-training limits must be non-negative integer counts")
    if until_epoch is not None:
        requested = until_epoch * steps_per_epoch
    elif additional_epochs is not None:
        if start_step % steps_per_epoch:
            raise ValueError("additional_epochs requires an epoch-boundary checkpoint")
        requested = start_step + additional_epochs * steps_per_epoch
    elif max_steps is not None:
        requested = start_step + max_steps
    else:
        requested = configured_steps
    if requested > configured_steps and (until_epoch is not None or additional_epochs is not None):
        raise ValueError("epoch limit exceeds the immutable configured training budget")
    return min(configured_steps, requested)


def post_training_steps_per_epoch(config, dataset_length):
    optimization = config["optimization"]
    explicit = optimization.get("steps_per_epoch")
    if explicit is not None:
        if isinstance(explicit, bool) or not isinstance(explicit, int) or explicit < 1:
            raise ValueError("optimization.steps_per_epoch must be a positive integer")
        if optimization.get("sampling") != "full_pass":
            raise ValueError("explicit steps_per_epoch requires continuous full_pass sampling")
        return explicit
    return max(
        1,
        math.ceil(
            dataset_length
            * float(optimization.get("train_fraction", 1.0))
            / int(optimization["batch_size"])
        ),
    )


def sample_exposure(completed_steps, dataset_length, batch_size):
    """Count real samples, including a short final batch on each complete pass."""
    batches_per_pass = math.ceil(dataset_length / batch_size)
    complete, remainder = divmod(completed_steps, batches_per_pass)
    samples = complete * dataset_length + min(remainder * batch_size, dataset_length)
    return {
        "samples_seen": samples,
        "dataset_passes": samples / dataset_length,
        "batches_per_dataset_pass": batches_per_pass,
    }
