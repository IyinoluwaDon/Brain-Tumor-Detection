# Brain MRI Image Classification

A Flask application that classifies uploaded brain MRI images using a compact logistic-regression baseline implemented with bundled NumPy model parameters. The shipped model is a replacement baseline for an earlier CNN whose checkpoint is not present in this repository.

Images are converted to grayscale and resized to 32 x 32 before inference. `scripts/train_baseline.py` trains the baseline from a locally supplied dataset archive; `models/` contains the model and evaluation metadata. The saved evaluation record reports **81.17% image-level accuracy on 786 held-out images**.

## Run locally

```sh
python -m pip install -r requirements.txt
python -m app.app
```

The dataset archive is supplied separately to the training script. `tests/` covers upload handling and inference through the bundled model.
