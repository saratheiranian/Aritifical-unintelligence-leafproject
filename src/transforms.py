"""Image transforms shared by training, evaluation and the Gradio app.

Kept free of the `datasets` dependency so the app can import it on its own.
"""

from torchvision import transforms

# ImageNet statistics: fine for a model trained from scratch and required for pretrained ResNets.
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def get_transforms(img_size=128, augment=False):
    """Resize + normalise for every split; random augmentation only for training."""
    steps = [transforms.Resize((img_size, img_size))]
    if augment:
        steps += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ]
    steps += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    return transforms.Compose(steps)
