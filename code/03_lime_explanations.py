"""
Generate local LIME explanations for N examples and save as CSV + PNGs.
Outputs in results/lime/
"""
from pathlib import Path
import joblib, pandas as pd, numpy as np
from lime.lime_text import LimeTextExplainer
from sklearn.pipeline import make_pipeline
from 00_config import DATA_PROCESSED, RESULTS, MODELS

OUT = RESULTS / "lime"
OUT.mkdir(parents=True, exist_ok=True)

def main(n_examples=20, seed=42):
    vec = joblib.load(MODELS / "vectorizer.joblib")
    clf = joblib.load(MODELS / "model.joblib")
    pipe = make_pipeline(vec, clf)

    test = pd.read_csv(DATA_PROCESSED / "test.csv").sample(n=min(n_examples, 200), random_state=seed)
    explainer = LimeTextExplainer(class_names=["neg","pos"])
    rows = []
    for i, row in test.reset_index(drop=True).iterrows():
        exp = explainer.explain_instance(row["text"], pipe.predict_proba, num_features=10)
        rows.append({
            "i": i,
            "text": row["text"],
            "pred": int(pipe.predict([row["text"]])[0]),
            "weights": dict(exp.as_list())
        })
        exp.save_to_file(str(OUT / f"lime_{i}.html"))
    pd.DataFrame(rows).to_json(OUT / "lime_explanations.json", orient="records", lines=False)
    print("Saved LIME explanations to", OUT)

if __name__ == "__main__":
    main()

