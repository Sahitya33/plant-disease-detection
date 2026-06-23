"""
The app package. We use Flask's "application factory" pattern
(create_app function) instead of a single global app object.

Why: it makes testing easier (tests/test_routes.py creates a fresh app
with test config) and avoids circular-import headaches as the project grows.
"""

import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()  # the database extension, created here, attached to an app below


def create_app(config_object="config.Config"):
    app = Flask(__name__)
    app.config.from_object(config_object)

    # Make sure the instance folder (where predictions.db lives) exists
    os.makedirs(os.path.join(app.root_path, "..", "instance"), exist_ok=True)
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

    db.init_app(app)

    from app.routes import main_bp
    app.register_blueprint(main_bp)

    with app.app_context():
        db.create_all()  # creates tables from models.py if they don't exist yet

    return app
