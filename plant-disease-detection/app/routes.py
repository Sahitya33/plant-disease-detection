import os
from flask import (
    Blueprint, current_app, render_template, request, redirect, url_for, flash
)
from werkzeug.utils import secure_filename

from app import db
from app.models import Prediction
from app.ml_model import predict_image

main_bp = Blueprint("main", __name__)


def _allowed_file(filename):
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return ext in current_app.config["ALLOWED_EXTENSIONS"]


@main_bp.route("/")
def index():
    # Query: get the 10 most recent predictions, newest first.
    # This is a real SQL query, written in Python via the ORM:
    #   SELECT * FROM predictions ORDER BY created_at DESC LIMIT 10;
    recent = (
        Prediction.query.order_by(Prediction.created_at.desc()).limit(10).all()
    )
    return render_template("index.html", recent=recent)


@main_bp.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("image")

    if file is None or file.filename == "":
        flash("Please choose an image to upload.")
        return redirect(url_for("main.index"))

    if not _allowed_file(file.filename):
        flash("Only .png, .jpg, .jpeg files are allowed.")
        return redirect(url_for("main.index"))

    filename = secure_filename(file.filename)
    save_path = os.path.join(current_app.config["UPLOAD_FOLDER"], filename)
    file.save(save_path)

    try:
        predicted_class, confidence = predict_image(
            save_path,
            model_path=current_app.config["MODEL_PATH"],
            class_names_path=current_app.config["CLASS_NAMES_PATH"],
            image_size=current_app.config["IMAGE_SIZE"],
        )
    except FileNotFoundError as e:
        flash(str(e))
        return redirect(url_for("main.index"))

    # --- This is the "write to database" step ---
    new_prediction = Prediction(
        filename=filename,
        predicted_class=predicted_class,
        confidence=confidence,
    )
    db.session.add(new_prediction)   # stage the new row
    db.session.commit()              # write it to predictions.db
    # ----------------------------------------------

    return render_template(
        "index.html",
        result=new_prediction,
        recent=Prediction.query.order_by(Prediction.created_at.desc()).limit(10).all(),
    )
