"""Keep reporting epochs separate from the uninterrupted shuffled data stream."""

import math


def base_training_steps_per_epoch(config, dataset_length):
    """Allow a fixed update budget while sampling the complete training split.

    Base training draws independent random batches, so an epoch is a reporting
    interval, not a guarantee of one complete dataset pass. The default keeps
    the historical interval and RNG behavior unchanged.
    """
    explicit = config["optimization"].get("steps_per_epoch")
    if explicit is not None:
        if isinstance(explicit, bool) or not isinstance(explicit, int) or explicit < 1:
            raise ValueError("optimization.steps_per_epoch must be a positive integer")
        return explicit
    return max(1, math.ceil(dataset_length / int(config["optimization"].get("batch_size", 1))))


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
