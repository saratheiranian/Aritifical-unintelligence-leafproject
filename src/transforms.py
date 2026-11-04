"""Image transforms shared by training, evaluation and the Gradio app.

Kept free of the `datasets` dependency so the app can import it on its own.
"""

from torchvision import transforms

# ImageNet statistics: fine for a model trained from scratch and required for pretrained ResNets.
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


AUGMENT_LEVELS = ("none", "basic", "strong")


def augmentation_steps(level, img_size=128):
    """Random training-only transforms for an augmentation level.

    basic: flips, small rotations and mild colour changes.
    strong: also random crops, vertical flips, larger rotations, stronger brightness and contrast
    changes and occasional blur. Mean brightness differs a lot between classes in PlantVillage
    (91 to 148 on a 0-255 scale), so strong colour jitter stops the model from using lighting as a
    shortcut, and crops and blur make lab photos look more like phone photos.
    """
    if level in (False, None, "none"):
        return []
    if level in (True, "basic"):
        return [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ]
    if level == "strong":
        return [
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.05),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.3),
        ]
    raise ValueError(f"augment must be one of {AUGMENT_LEVELS}, got {level!r}")


def get_transforms(img_size=128, augment=False):
    """Resize + normalise for every split; random augmentation only for training.

    augment: "none", "basic" or "strong" (True means "basic", False means "none").
    """
    steps = [transforms.Resize((img_size, img_size))]
    steps += augmentation_steps(augment, img_size)
    steps += [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    return transforms.Compose(steps)
