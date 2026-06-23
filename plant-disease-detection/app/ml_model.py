"""
Loads the trained model once and exposes a predict_image() function.

Why a separate module: routes.py shouldn't need to know HOW predictions
are made (Keras vs PyTorch vs a remote API). It just calls predict_image()
and gets back (class_name, confidence). This separation is good practice -
you could swap the model implementation later without touching the web layer.
"""

import json
import os
import numpy as np
from PIL import Image

_model = None
_class_names = None


def _load_model_if_needed(model_path, class_names_path):
    global _model, _class_names
    if _model is None:
        import tensorflow as tf  # imported lazily so the app can start without TF installed yet

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"No trained model found at {model_path}.\n"
                "Run model_training/train.py first (see model_training/README.md)."
            )
        _model = tf.keras.models.load_model(model_path)

        with open(class_names_path) as f:
            _class_names = json.load(f)

    return _model, _class_names


def predict_image(image_path, model_path, class_names_path, image_size=(224, 224)):
    """
    Returns: (predicted_class: str, confidence: float between 0 and 1)
    """
    model, class_names = _load_model_if_needed(model_path, class_names_path)

    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    img = Image.open(image_path).convert("RGB").resize(image_size)
    arr = np.array(img).astype("float32")
    # Must match training preprocessing exactly (see model_training/train.py).
    # Training and inference preprocessing mismatches are a common source
    # of "the model works in training but predicts garbage in the app" bugs.
    arr = preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)  # model expects a batch dimension

    predictions = model.predict(arr, verbose=0)[0]
    best_index = int(np.argmax(predictions))

    predicted_class = class_names[best_index]
    confidence = float(predictions[best_index])

    return predicted_class, confidence
