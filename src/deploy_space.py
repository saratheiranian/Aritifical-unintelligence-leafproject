"""Deploy the Gradio app to a public Hugging Face Space.

    huggingface-cli login            # once, with a token that has write access
    python -m src.make_app_assets    # optional: example gallery + class examples
    python -m src.deploy_space --space-id <hf-user>/plant-disease-classifier \
        --checkpoint outputs/best_model.pt

Instead of uploading checkpoints, you can point the Space at a model repo made with
publish_model.py:  --hf-model-repo <hf-user>/plant-disease-cnn

The Space gets: app.py, the src/ modules the app needs, requirements.txt, README.md (Space
config), models/*.pt, examples/ and class_examples/.
"""

import argparse
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
APP_SRC_FILES = ["__init__.py", "models.py", "transforms.py", "inference.py"]

SPACE_README = """---
title: Plant Disease Classifier
emoji: 🌿
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: {gradio_version}
app_file: app.py
pinned: false
---

Upload a leaf photo to get the 5 most likely plant diseases (39 PlantVillage classes).
KCL 5CCSAGAP small group project. Not a diagnostic tool.
"""


def stage(out, checkpoints, gradio_version):
    """Copy everything the Space needs into `out`."""
    shutil.copy(ROOT / "app" / "app.py", out / "app.py")
    shutil.copy(ROOT / "app" / "requirements.txt", out / "requirements.txt")
    (out / "src").mkdir()
    for name in APP_SRC_FILES:
        shutil.copy(ROOT / "src" / name, out / "src" / name)
    (out / "README.md").write_text(SPACE_README.format(gradio_version=gradio_version))
    for folder in ["examples", "class_examples"]:
        if (ROOT / "app" / folder).is_dir():
            shutil.copytree(ROOT / "app" / folder, out / folder)
    if checkpoints:
        (out / "models").mkdir()
        for ckpt in checkpoints:
            shutil.copy(ckpt, out / "models" / Path(ckpt).name)
    return sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file())


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--space-id", required=True)
    p.add_argument("--checkpoint", nargs="*", default=[], help="checkpoints to upload into models/")
    p.add_argument("--hf-model-repo", default=None, help="load models from this HF model repo instead")
    p.add_argument("--dry-run", action="store_true", help="only list the files that would be uploaded")
    args = p.parse_args(argv)
    if not args.checkpoint and not args.hf_model_repo:
        p.error("give --checkpoint and/or --hf-model-repo")

    import gradio
    from huggingface_hub import HfApi

    with tempfile.TemporaryDirectory() as tmp:
        files = stage(Path(tmp), args.checkpoint, gradio.__version__)
        print("\n".join(files))
        if args.dry_run:
            return
        api = HfApi()
        api.create_repo(args.space_id, repo_type="space", space_sdk="gradio", exist_ok=True)
        if args.hf_model_repo:
            api.add_space_variable(args.space_id, "HF_MODEL_REPO", args.hf_model_repo)
        api.upload_folder(folder_path=tmp, repo_id=args.space_id, repo_type="space",
                          commit_message="Deploy app")
    print(f"Deployed to https://huggingface.co/spaces/{args.space_id}")


if __name__ == "__main__":
    main()
