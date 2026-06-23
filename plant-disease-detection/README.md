# 🌿 Plant Disease Detector

A web app that takes a photo of a plant leaf, predicts whether (and what)
disease it has using a CNN, and keeps a history of predictions in a database.

Built as a learning project covering: **Flask web apps**, **Git/GitHub**,
and **SQL databases via SQLAlchemy**.

## Stack
- **ML**: TensorFlow/Keras, transfer learning on MobileNetV2, trained on
  the [PlantVillage dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
- **Backend**: Flask
- **Database**: SQLite (via SQLAlchemy ORM — upgradeable to PostgreSQL later)
- **Frontend**: Server-rendered HTML (Jinja2) + plain CSS

## Project structure
```
plant-disease-detection/
├── app/
│   ├── __init__.py        # Flask app factory, registers DB + routes
│   ├── models.py           # Database schema (Prediction table)
│   ├── routes.py           # Web routes: upload, predict, history
│   ├── ml_model.py         # Loads trained model, runs inference
│   ├── templates/          # HTML pages
│   └── static/              # CSS + uploaded images
├── model_training/
│   ├── train.py            # Trains the CNN
│   ├── data/                # (gitignored) put the dataset here
│   └── README.md
├── tests/
│   └── test_routes.py
├── docs/
│   └── GIT_WORKFLOW.md     # Step-by-step git/GitHub guide for this project
├── config.py
├── run.py
├── requirements.txt
└── .gitignore
```

## Setup

1. **Clone and enter the project** (or just `cd` into this folder if you
   downloaded it directly — see `docs/GIT_WORKFLOW.md` to turn it into a
   real git repo and push it to GitHub).

2. **Create a virtual environment** (keeps this project's packages
   separate from everything else on your machine):
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # macOS/Linux:
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the model** (see `model_training/README.md` for dataset setup):
   ```bash
   python model_training/train.py
   ```

5. **Run the app:**
   ```bash
   python run.py
   ```
   Visit http://127.0.0.1:5000

## How the database fits in

Every time you upload an image and get a prediction, a row is written to
`instance/predictions.db` (a SQLite file created automatically on first
run). The schema is defined in `app/models.py` — open that file first to
understand the DB side of this project. The home page queries that table
to show your last 10 predictions.

To poke around the database directly (good for learning SQL):
```bash
sqlite3 instance/predictions.db
sqlite> .tables
sqlite> SELECT * FROM predictions ORDER BY created_at DESC LIMIT 5;
sqlite> .schema predictions
```

## Learning roadmap (suggested order)

1. Get the app running with a placeholder/dummy model so you see the
   Flask + DB plumbing work end-to-end.
2. Read `docs/GIT_WORKFLOW.md` and turn this folder into a real git repo,
   make your first few commits.
3. Train the real model (`model_training/`).
4. Push to GitHub, and practice the branch → commit → PR → merge cycle
   by adding one small feature yourself (ideas below).
5. (Optional) Deploy somewhere like Render or Railway.

### Feature ideas to practice Git branching
Each of these is a good "create a branch, build it, merge it" exercise:
- Add a delete button for history entries (DELETE in SQL/SQLAlchemy)
- Add a confidence threshold — flag low-confidence predictions as "uncertain"
- Add pagination to the history table instead of just showing 10
- Add a simple `/api/predictions` JSON endpoint
