# Deployment notes (role 5)

## Hugging Face account (task 3, 10 Nov)

- Hugging Face user: `Media1234`
- Logged in with `hf auth login` and a write token. The token is never committed.
- With huggingface_hub 2.x the login command is `hf auth login`; older versions used `huggingface-cli login`.
- On Windows PowerShell, activating the venv may first need
  `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

## Dry run (task 4, 10 Nov)

    python -m src.deploy_space --space-id Media1234/plant-disease-classifier --hf-model-repo Media1234/plant-disease-cnn --dry-run

Files that would be uploaded to the Space:

- `README.md` (Space settings)
- `app.py`
- `requirements.txt`
- `src/__init__.py`, `src/inference.py`, `src/models.py`, `src/transforms.py`

`examples/` and `class_examples/` are added after running `python -m src.make_app_assets`.

## Local test (task 6, 10 Nov)

- `python app/app.py --checkpoint outputs/final_model.pt` runs with Gradio 6.31 and the 99% model.
- A test leaf photo gave "Corn: Common rust" at 73%, with the top 5 chart and probability table.
- The "training images of the predicted class" gallery is empty until `python -m src.make_app_assets` is run (task 2).

## Model upload (task 6, 10 Nov)

- `python -m src.publish_model --repo-id Media1234/plant-disease-cnn --checkpoint outputs/final_model.pt`
- The model is on the Hugging Face Hub as `cnn.pt`: https://huggingface.co/Media1234/plant-disease-cnn

## Space creation blocked (10 Nov)

- The full deploy command:

      python -m src.deploy_space --space-id Media1234/plant-disease-classifier --hf-model-repo Media1234/plant-disease-cnn

  failed with `402 Payment Required`: "Static Spaces are free for everyone,
  but hosting Gradio and Docker Spaces on free cpu-basic requires a PRO subscription."
- The Hugging Face documentation says that Gradio and Docker Spaces require a paid plan to create,
  and that free personal accounts in good standing can still host up to 2 Gradio Spaces on ZeroGPU
  (source: https://huggingface.co/docs/hub/en/spaces-overview). We have not tried the ZeroGPU option.
- The lecturer confirmed the problem on the KEATS Q&A forum ("Gradio tutorial"): do not buy a subscription,
  keep testing locally or in Colab, and the teaching team will clarify the deployment arrangements.

## CPU timing (issue #32, measured 10 Nov)

Measured on a Windows laptop CPU (Snapdragon X 12-core X1E80100 @ 3.40 GHz, Python 3.11,
PyTorch 2.15.0, CPU build), with `outputs/final_model.pt` and one leaf photo, using `Predictor`
from `src/inference.py`:

- Model load: 0.07 s (once, when the app starts)
- First prediction: 0.129 s
- Average over 20 predictions: 0.007 s

So the requirement of under 2 s per prediction on a laptop CPU holds. These times are for the
model only; the app adds a little time for uploading the image and showing the results.
Maryam's MacBook (Apple M4, Python 3.12, PyTorch 2.15) gave 0.013 s, 0.023 s and 0.002 s.

Command used (run from the repo root, with a leaf photo saved as `outputs/leaf.jpg`):

    python -c "import time; from PIL import Image; from src.inference import Predictor; t0=time.perf_counter(); p=Predictor('outputs/final_model.pt', device='cpu'); t1=time.perf_counter(); img=Image.open('outputs/leaf.jpg').convert('RGB'); p.top_k(img,5); t2=time.perf_counter(); [p.top_k(img,5) for _ in range(20)]; t3=time.perf_counter(); print(f'Model load: {t1-t0:.2f} s'); print(f'First prediction: {t2-t1:.3f} s'); print(f'Average of 20 predictions: {(t3-t2)/20:.3f} s')"

## Next

- Deploy the Space once the teaching team confirms the deployment option.