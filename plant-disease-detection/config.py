"""
Central configuration for the app.

Why this file exists (DB learning note):
Instead of hardcoding things like the database location inside our code,
we keep them here. Later, if you move from SQLite to PostgreSQL, you only
change ONE line (SQLALCHEMY_DATABASE_URI) and nothing else in the app
needs to change. That's the whole point of using an ORM (SQLAlchemy).
"""

import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


class Config:
    # --- Database ---
    # SQLite stores the whole database as a single file on disk.
    # The 'instance' folder is Flask's conventional place for files that
    # shouldn't be committed to git (see .gitignore).
    SQLALCHEMY_DATABASE_URI = (
        f"sqlite:///{os.path.join(BASE_DIR, 'instance', 'predictions.db')}"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False  # turns off a feature we don't need

    # --- File uploads ---
    UPLOAD_FOLDER = os.path.join(BASE_DIR, "app", "static", "uploads")
    MAX_CONTENT_LENGTH = 5 * 1024 * 1024  # 5 MB max upload size
    ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

    # --- ML model ---
    MODEL_PATH = os.path.join(BASE_DIR, "model_training", "plant_disease_model.h5")
    CLASS_NAMES_PATH = os.path.join(BASE_DIR, "model_training", "class_names.json")
    IMAGE_SIZE = (224, 224)

    # --- Flask secret (used for sessions/flash messages) ---
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-change-this-in-production")
