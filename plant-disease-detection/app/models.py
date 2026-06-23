"""
Database models.

This is the heart of the "learn databases" part of the project.
Each class below becomes a TABLE. Each attribute becomes a COLUMN.
SQLAlchemy (the ORM) translates Python objects <-> SQL rows for you,
so you write Python, not raw SQL, while still learning DB concepts:
primary keys, columns/types, and querying.
"""

from datetime import datetime
from app import db


class Prediction(db.Model):
    """
    One row = one image the user uploaded and the model's prediction for it.
    This is what lets the app show "prediction history".
    """

    __tablename__ = "predictions"

    # Primary key: a unique ID auto-generated for every row (1, 2, 3, ...)
    id = db.Column(db.Integer, primary_key=True)

    # The saved filename of the uploaded image
    filename = db.Column(db.String(255), nullable=False)

    # The model's predicted class, e.g. "Tomato___Late_blight"
    predicted_class = db.Column(db.String(100), nullable=False)

    # The model's confidence, e.g. 0.9421 (= 94.21%)
    confidence = db.Column(db.Float, nullable=False)

    # When the prediction was made, set automatically
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<Prediction {self.id}: {self.predicted_class} ({self.confidence:.2%})>"

    def to_dict(self):
        """Handy for turning a row into JSON if you build an API later."""
        return {
            "id": self.id,
            "filename": self.filename,
            "predicted_class": self.predicted_class,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat(),
        }
