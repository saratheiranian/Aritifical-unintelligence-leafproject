import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models import PlantCNN
from src.train import EarlyStopping, make_scheduler, parse_args, run_epoch


def test_early_stopping_waits_for_patience():
    stopper = EarlyStopping(patience=2, min_delta=0.01)
    assert stopper.step(0.50) is True
    assert stopper.step(0.505) is False  # below min_delta
    assert not stopper.should_stop
    assert stopper.step(0.49) is False
    assert stopper.should_stop
    assert stopper.best == 0.50


def test_early_stopping_resets_on_improvement():
    stopper = EarlyStopping(patience=2)
    stopper.step(0.5)
    stopper.step(0.4)
    assert stopper.step(0.6) is True
    assert stopper.epochs_without_gain == 0


def test_run_epoch_trains_and_evaluates():
    torch.manual_seed(0)
    data = TensorDataset(torch.randn(16, 3, 32, 32), torch.randint(0, 4, (16,)))
    loader = DataLoader(data, batch_size=8)
    model = PlantCNN(num_classes=4, base_channels=4, depth=2)
    before = [p.clone() for p in model.parameters()]
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    loss, acc = run_epoch(model, loader, nn.CrossEntropyLoss(), "cpu", optimizer)
    assert loss > 0 and 0.0 <= acc <= 1.0
    assert any(not torch.equal(a, b) for a, b in zip(before, model.parameters()))

    after = [p.clone() for p in model.parameters()]
    run_epoch(model, loader, nn.CrossEntropyLoss(), "cpu")
    assert all(torch.equal(a, b) for a, b in zip(after, model.parameters()))
    assert not model.training


def test_schedulers():
    optimizer = torch.optim.SGD([nn.Parameter(torch.zeros(1))], lr=0.1)
    assert make_scheduler("none", optimizer, 10) is None
    assert make_scheduler("cosine", optimizer, 10) is not None
    assert make_scheduler("plateau", optimizer, 10) is not None


def test_default_args():
    args = parse_args([])
    assert args.arch == "cnn" and args.batch_size == 64 and args.img_size == 128
