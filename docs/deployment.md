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

## Next

- Task 6: publish the model and deploy the Space once `final_model.pt` is available.