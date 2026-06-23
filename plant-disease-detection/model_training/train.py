"""
Trains a leaf-disease classifier using transfer learning on MobileNetV2.

Expected folder layout (this is how Kaggle's PlantVillage dataset is
already organized, one folder per class):

    model_training/data/
        Apple___Black_rot/
            img1.jpg
            img2.jpg
        Apple___healthy/
            ...
        Tomato___Late_blight/
            ...

Run from the project root:
    python model_training/train.py
"""

import json
import os

import tensorflow as tf

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
MODEL_OUT = os.path.join(os.path.dirname(__file__), "plant_disease_model.h5")
CLASS_NAMES_OUT = os.path.join(os.path.dirname(__file__), "class_names.json")


def build_datasets():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
    )
    class_names = train_ds.class_names

    # IMPORTANT: MobileNetV2 was pretrained on ImageNet using a specific
    # input scaling ([-1, 1], not the more generic [0, 1]). We must use
    # its own preprocess_input so the pretrained features behave the way
    # they were trained to. Using the wrong scaling silently hurts accuracy
    # rather than throwing an error, which makes it an easy bug to miss.
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

    # Data augmentation: PlantVillage images are all shot in controlled lab
    # conditions against a plain background. Without augmentation, a model
    # can partly learn "background/lighting" instead of "disease pattern"
    # and generalize poorly to real-world photos. These layers only apply
    # during training (have no effect at inference/prediction time).
    augment = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.15),
        tf.keras.layers.RandomZoom(0.15),
        tf.keras.layers.RandomContrast(0.15),
    ])

    train_ds = (
        train_ds
        .map(lambda x, y: (augment(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
        .map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )
    # Validation data is NOT augmented — augmentation is a training-time
    # trick only. Validation should reflect real, unmodified images so the
    # accuracy you see is a trustworthy estimate.
    val_ds = (
        val_ds
        .map(lambda x, y: (preprocess_input(x), y), num_parallel_calls=tf.data.AUTOTUNE)
        .prefetch(tf.data.AUTOTUNE)
    )

    return train_ds, val_ds, class_names


def build_model(num_classes):
    base = tf.keras.applications.MobileNetV2(
        input_shape=IMAGE_SIZE + (3,), include_top=False, weights="imagenet"
    )
    base.trainable = False  # freeze pretrained weights — we only train the new head

    model = tf.keras.Sequential([
        base,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    if not os.path.isdir(DATA_DIR):
        raise SystemExit(
            f"No data found at {DATA_DIR}.\n"
            "Download the PlantVillage dataset and place class folders inside "
            "model_training/data/. See model_training/README.md."
        )

    train_ds, val_ds, class_names = build_datasets()
    print(f"Found {len(class_names)} classes: {class_names[:5]}...")

    model = build_model(num_classes=len(class_names))
    model.summary()

    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)

    model.save(MODEL_OUT)
    with open(CLASS_NAMES_OUT, "w") as f:
        json.dump(class_names, f, indent=2)

    print(f"\nSaved model to {MODEL_OUT}")
    print(f"Saved class names to {CLASS_NAMES_OUT}")


if __name__ == "__main__":
    main()
/