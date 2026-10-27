"""Load a trained checkpoint and predict top-k diseases for one image.

Used by the Gradio app; kept free of Gradio and `datasets` so it is cheap to import and test.
"""

import torch

from src.models import build_model
from src.transforms import get_transforms


def pretty_name(name):
    """'Tomato___Late_blight' -> 'Tomato: Late blight'."""
    return name.replace("___", ": ").replace("_", " ").replace("  ", " ").strip()


class Predictor:
    def __init__(self, checkpoint_path, device="cpu"):
        ckpt = torch.load(checkpoint_path, map_location=device)
        self.model = build_model(ckpt["arch"], **ckpt["model_kwargs"])
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(device).eval()
        self.device = device
        self.class_names = ckpt["class_names"]
        self.img_size = ckpt["img_size"]
        self.arch = ckpt["arch"]
        self.val_acc = ckpt.get("val_acc")
        self.transform = get_transforms(self.img_size, augment=False)

    @torch.no_grad()
    def predict_proba(self, images):
        """Class probabilities for a list of PIL images, shape (n, num_classes)."""
        batch = torch.stack([self.transform(img.convert("RGB")) for img in images]).to(self.device)
        return torch.softmax(self.model(batch), dim=1).cpu()

    def top_k(self, image, k=5):
        """[(class_name, probability), ...] for the k most likely classes, highest first."""
        probs = self.predict_proba([image])[0]
        values, indices = torch.topk(probs, min(k, len(self.class_names)))
        return [(self.class_names[i], float(p)) for p, i in zip(values, indices)]
