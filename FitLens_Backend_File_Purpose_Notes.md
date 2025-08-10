# 📂 FitLens Backend – File Purpose Notes

## 1. `config.py`

Stores configuration settings such as the **Supabase URL** and **secret API key**.  
Reads these values from `.env` so that sensitive credentials are not hardcoded.

## 2. `db.py`

Initializes and returns a Supabase client connection.  
This allows other modules to interact with the database without repeating connection logic.

## 3. `main.py`

The main FastAPI application file.  
Defines the backend’s API endpoints:

- **`/analyze-and-recommend`** → Accepts a photo and options (style, size, budget) and returns product recommendations.
- **`/track`** → Records user interactions (clicks, likes) for analytics.  
  Also contains an MVP helper function for extracting simple traits from the uploaded image.

## 4. `models.py`

Defines **Pydantic models** for data validation and structure.  
Ensures incoming requests have the right data format and outgoing responses follow a consistent schema.

## 5. `ml/` (Machine Learning Module)

Holds files related to the **ML-based scoring system**.

- **`__init__.py`** → Marks the `ml` folder as a Python package.
- **`ml_scorer.py`** → Loads a trained ML model and scores products based on learned patterns from user interaction data. Replaces `scoring.py` when ML is enabled.
- **`train_model.py`** → Script for training the ML model. Reads interaction data, extracts features, trains a classifier (Logistic Regression or similar), and saves it for use in `ml_scorer.py`.
