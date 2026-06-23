# Model Training

## 1. Get the dataset
Download the PlantVillage dataset from Kaggle:
https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset

(Free Kaggle account needed. Alternative mirror: `emmarex/plantdisease`.)

It contains ~54,000 images across 38 classes (combinations of 14 crop
species x disease/healthy).

## 2. Arrange it
Unzip it so you end up with:

```
model_training/data/
    Apple___Apple_scab/
    Apple___Black_rot/
    Apple___healthy/
    Tomato___Late_blight/
    ... (38 folders total)
```

This folder is in `.gitignore` — datasets are too large and don't belong
in git history. Each teammate/computer downloads it separately.

## Preprocessing — what's automatic vs what you might add

Handled automatically by `train.py`:
- Resizing every image to 224×224
- Reading class labels from folder names
- Train/validation split (80/20)
- Correct input scaling for MobileNetV2 (`preprocess_input`)
- Data augmentation on the training set (flip/rotate/zoom/contrast) —
  important for PlantVillage specifically, since its images are all shot
  in controlled lab conditions; augmentation reduces (not eliminates) the
  risk of the model leaning on background/lighting cues instead of the
  actual leaf disease pattern.

Worth doing manually, optional, not required to get a working model:
- **Check for corrupt/non-image files** before training — a single bad
  file can crash `image_dataset_from_directory`. Quick check:
  ```python
  from PIL import Image
  import pathlib
  for p in pathlib.Path("data").rglob("*.*"):
      try:
          Image.open(p).verify()
      except Exception as e:
          print("Bad file:", p, e)
  ```
- **Class imbalance** — some PlantVillage classes have far more images
  than others. The model above doesn't correct for this. If you notice
  the model is much worse on rare classes, look into `class_weight` in
  `model.fit()` or oversampling.
- **Held-out real-world test images** — since PlantVillage is lab-condition
  data, it's worth manually testing the finished app on a few photos you
  take yourself (different background/lighting) to see how it actually
  generalizes, rather than trusting validation accuracy alone.


## 3. Train
```bash
pip install -r ../requirements.txt
python train.py
```

This uses transfer learning (MobileNetV2 pretrained on ImageNet, with a
new classification head trained on your data) — much faster and more
accurate than training a CNN from scratch on this dataset size. On a
laptop CPU expect ~10-20 min/epoch; a GPU (e.g. Colab) is much faster.

This produces:
- `plant_disease_model.h5` — the trained model
- `class_names.json` — maps prediction indices to class names

Both are gitignored (also too large / regeneratable). Run this once
locally and the Flask app will pick them up automatically.

## 5. Improving it later
- Unfreeze the last few layers of MobileNetV2 and fine-tune with a low
  learning rate for a few more epochs.
- Track experiments (which hyperparameters → which accuracy) in a simple
  table or with a tool like MLflow once you're comfortable with the basics.
