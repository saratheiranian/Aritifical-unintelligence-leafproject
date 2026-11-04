from types import SimpleNamespace

import pytest
import torch
from datasets import ClassLabel, Dataset
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

from src.data import LeafDataset
from src.inference import Predictor
from src.models import build_model
from src.train import main as train_main
from src.train import run_epoch, save_checkpoint
from src.transforms import get_transforms

NUM_CLASSES = 3
IMG_SIZE = 64
CLASS_NAMES = [f"Class{i}" for i in range(NUM_CLASSES)]
# One clearly separable base colour per class, so the fake data is actually learnable.
BASE_COLORS = [(200, 40, 40), (40, 200, 40), (40, 40, 200)]
MODEL_KWARGS = {
    "num_classes": NUM_CLASSES,
    "base_channels": 4,
    "depth": 2,
    "dropout": 0.0,
    "use_batchnorm": True,
}


def make_split(n_per_class, shift):
    """Solid-colour images, one colour family per class, with a small per-image shift (clamped to 255)."""
    images, labels = [], []
    for i in range(n_per_class):
        for label, base in enumerate(BASE_COLORS):
            color = tuple(min(255, c + shift + 3 * i) for c in base)
            images.append(Image.new("RGB", (IMG_SIZE, IMG_SIZE), color))
            labels.append(label)
    return images, labels


def make_hf_dataset(images, labels):
    ds = Dataset.from_dict({"image": images, "label": labels})
    return ds.cast_column("label", ClassLabel(names=CLASS_NAMES))


def make_loaders(batch_size=8):
    """Tiny train/val/test loaders: 24 / 9 / 9 images, 3 classes."""
    transform = get_transforms(IMG_SIZE, augment=False)
    specs = {"train": (8, 0, True), "val": (3, 5, False), "test": (3, 10, False)}
    loaders = {}
    for name, (n_per_class, shift, shuffle) in specs.items():
        ds = make_hf_dataset(*make_split(n_per_class, shift))
        loaders[name] = DataLoader(
            LeafDataset(ds, "image", "label", transform), batch_size=batch_size, shuffle=shuffle
        )
    return loaders


def test_train_evaluate_predict_pipeline(tmp_path):
    torch.manual_seed(42)
    device = "cpu"
    loaders = make_loaders()

    # Same construction path as train.py and Predictor, so saved kwargs always match the model.
    model = build_model("cnn", **MODEL_KWARGS).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    # Loss before any training (evaluation mode, no optimizer).
    initial_loss, _ = run_epoch(model, loaders["train"], criterion, device)

    # Enough steps for BatchNorm running statistics to settle before evaluating.
    for epoch in range(1, 9):
        train_loss, _ = run_epoch(model, loaders["train"], criterion, device, optimizer)
        _, val_acc = run_epoch(model, loaders["val"], criterion, device)

    assert train_loss < initial_loss, "Training should reduce the loss"

    # Save a checkpoint in the same format as train.py.
    checkpoint_path = tmp_path / "model.pt"
    args = SimpleNamespace(arch="cnn", img_size=IMG_SIZE)
    save_checkpoint(checkpoint_path, model, args, MODEL_KWARGS, CLASS_NAMES, val_acc, epoch)
    assert checkpoint_path.exists()

    # Evaluate on the held-out test split.
    _, test_acc = run_epoch(model, loaders["test"], criterion, device)
    assert test_acc > 1 / NUM_CLASSES, f"Test accuracy {test_acc:.3f} should beat random guessing"

    # Load through Predictor and make predictions.
    predictor = Predictor(checkpoint_path, device=device)
    test_image = Image.new("RGB", (IMG_SIZE, IMG_SIZE), BASE_COLORS[0])
    predictions = predictor.top_k(test_image, k=NUM_CLASSES)

    assert len(predictions) == NUM_CLASSES
    assert all(name in CLASS_NAMES for name, _ in predictions)
    probs = [p for _, p in predictions]
    assert all(0.0 <= p <= 1.0 for p in probs)
    assert sum(probs) == pytest.approx(1.0, abs=1e-4)  # k == number of classes
    assert probs == sorted(probs, reverse=True)

    # Round trip: the loaded model must reproduce the trained model's output exactly.
    model.eval()
    transform = get_transforms(IMG_SIZE, augment=False)
    with torch.no_grad():
        expected = torch.softmax(model(transform(test_image).unsqueeze(0)), dim=1)[0]
    torch.testing.assert_close(predictor.predict_proba([test_image])[0], expected)

    ckpt = torch.load(checkpoint_path, map_location=device)
    assert ckpt["arch"] == "cnn"
    assert ckpt["class_names"] == CLASS_NAMES
    assert ckpt["img_size"] == IMG_SIZE


def test_train_main_writes_loadable_checkpoint(tmp_path, monkeypatch):
    """Runs train.main itself (no ClearML, fake loaders) and loads its best_model.pt."""
    torch.manual_seed(42)
    loaders = make_loaders()
    monkeypatch.setattr("src.train.make_loaders", lambda **kwargs: (loaders, CLASS_NAMES))

    best_val = train_main(
        [
            "--no-clearml",
            "--arch", "cnn",
            "--base-channels", "4",
            "--depth", "2",
            "--dropout", "0.0",
            "--epochs", "2",
            "--img-size", str(IMG_SIZE),
            "--output-dir", str(tmp_path),
        ]
    )

    best_path = tmp_path / "best_model.pt"
    assert best_path.exists()
    assert 0.0 <= best_val <= 1.0

    predictor = Predictor(best_path, device="cpu")
    predictions = predictor.top_k(Image.new("RGB", (IMG_SIZE, IMG_SIZE), BASE_COLORS[1]), k=2)
    assert len(predictions) == 2
    assert predictor.class_names == CLASS_NAMES