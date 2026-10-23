# Brain MRI Image Classifier — Educational Baseline

A Flask demo that classifies PNG/JPEG images with a small shipped logistic-regression baseline. It replaces an unavailable original CNN checkpoint; this is a distinct, deliberately simple experiment.

## What is included

- Actual upload-to-prediction path using the bundled NumPy model
- Grayscale 32 × 32 preprocessing
- Training script and evaluation metadata for the replacement baseline

## Run locally

```sh
python -m pip install -r requirements.txt
python -m app.app
```

## Verify

python -m pip install -r requirements-dev.txt
python -m pytest -q

## Repository map

`app/` contains the Flask demo. `models/` contains the compact baseline and its evaluation record. `scripts/train_baseline.py` reproduces training from a locally supplied archive; `tests/` exercises inference and uploads.

## Evidence and limitations

The bundled baseline has a recorded image-level held-out result of 81.17% accuracy on 786 images, and the tests include a real upload through the shipped model. The original CNN weights are absent.

Patient identifiers are unavailable; patient separation, duplicate-image effects and external/clinical validity are unresolved. This is not a diagnostic tool. Training data is not redistributed; review the source dataset's stated license and conditions before use.

## License

No repository license file is included. Unless the project owner adds a license, do not assume the code, data or model artifacts may be redistributed or reused. Verify third-party data and model terms separately.
