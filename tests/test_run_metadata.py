"""Tests for training run metadata helpers."""

from dualdet.utils.run_metadata import infer_best_epoch


def test_infer_best_epoch_from_history() -> None:
    history = [
        {"epoch": 1, "val": {"total": 3.0, "ap": 0.10}},
        {"epoch": 2, "val": {"total": 2.5, "ap": 0.25}},
        {"epoch": 3, "val": {"total": 2.7, "ap": 0.20}},
    ]
    best_epoch, best_metric = infer_best_epoch(history, metric="val.ap")
    assert best_epoch == 2
    assert best_metric == 0.25

    best_epoch, best_loss = infer_best_epoch(history, metric="val.total")
    assert best_epoch == 2
    assert best_loss == 2.5
