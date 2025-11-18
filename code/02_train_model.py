"""
Train TF-IDF + LogisticRegression on processed data.
Saves: models/vectorizer.joblib, models/model.joblib
"""
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib
from 00_config import DATA_PROCESSED, MODELS

MODELS.mkdir(parents=True, exist_ok=True)

def main():
    train = pd.read_csv(DATA_PROCESSED / "train.csv")
    test = pd.read_csv(DATA_PROCESSED / "test.csv")
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=20000, ngram_range=(1,2))),
        ("clf", LogisticRegression(max_iter=200, n_jobs=-1))
    ])
    pipe.fit(train["text"], train["label"])
    preds = pipe.predict(test["text"])
    print(classification_report(test["label"], preds, digits=3))
    # Save fitted objects
    # Separate saves so later LIME/SHAP can access tfidf vocabulary
    joblib.dump(pipe.named_steps["tfidf"], MODELS / "vectorizer.joblib")
    joblib.dump(pipe.named_steps["clf"], MODELS / "model.joblib")
    print("Saved vectorizer and model under", MODELS)

if __name__ == "__main__":
    main()

