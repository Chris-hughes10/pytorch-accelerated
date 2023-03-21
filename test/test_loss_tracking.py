import torch
from pytorch_accelerated.tracking import LossTracker


def test_can_track_loss_incrementally():
    tracker = LossTracker()
    losses = [105.5, 94.7, 98.0, 55.2, 32.5]
    expected_loss = sum(losses) / len(losses)

    for loss in losses:
        tracker.update(loss)
    calculated_loss = tracker.average

    assert expected_loss == calculated_loss


def test_can_track_loss_for_varying_batch_sizes():
    tracker = LossTracker()
    losses = [105.5, 94.7, 98.0, 55.2, 32.5]
    batch_sizes = [64, 32, 16, 8, 4]
    total_count = sum(batch_sizes)
    total_loss = (torch.tensor(losses) * torch.tensor(batch_sizes)).sum().item()
    expected_loss = total_loss / total_count

    for loss, batch_size in zip(losses, batch_sizes):
        tracker.update(loss, batch_size=batch_size)
    calculated_loss = tracker.average

    assert expected_loss == calculated_loss


def test_calculate_loss_average_calculation():
    tracker = LossTracker()
    losses = [105.5, 94.7, 98.0, 55.2, 32.5]
    batch_sizes = [64, 32, 16, 8, 4]
    total_count = sum(batch_sizes)
    total_loss = (torch.tensor(losses) * torch.tensor(batch_sizes)).sum().item()
    expected_loss = total_loss / total_count

    for loss, batch_size in zip(losses, batch_sizes):
        tracker.update(loss, batch_size=batch_size)
    calculated_loss = tracker.total_loss / tracker.running_count

    assert expected_loss == calculated_loss
