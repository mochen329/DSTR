"""Model-independent PyTorch training utilities."""

from itertools import islice

import torch


def train_one_epoch(model, loader, optimizer, loss_step, *, device, scaler,
                    accumulation_steps=1, max_grad_norm=None):
    """Train for one epoch and return the sample-averaged loss.

    The loader yields tuples of tensors with a shared batch dimension.
    loss_step(model, batch) returns a scalar loss averaged over the batch.
    The caller supplies the model, loss, optimizer, and AMP scaler.
    """
    if accumulation_steps < 1:
        raise ValueError("accumulation_steps must be positive")
    model.train()
    batches = iter(loader)
    total_samples, total_loss = 0, 0.0

    while group := list(islice(batches, accumulation_steps)):
        samples = sum(batch[0].shape[0] for batch in group)
        optimizer.zero_grad(set_to_none=True)
        for batch in group:
            size = batch[0].shape[0]
            batch = tuple(tensor.to(device, non_blocking=True) for tensor in batch)
            with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
                loss = loss_step(model, batch)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            scaler.scale(loss * (size / samples)).backward()
            total_loss += loss.detach().item() * size
            total_samples += size

        if max_grad_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_grad_norm, error_if_nonfinite=not scaler.is_enabled()
            )
        scaler.step(optimizer)
        scaler.update()

    if total_samples == 0:
        raise ValueError("The training loader did not yield any samples")
    return total_loss / total_samples
