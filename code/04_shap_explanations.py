"""
Generate SHAP explanations (LinearExplainer) and save per-example top tokens.
Outputs in results/shap/
"""
from pathlib import Path
import joblib, pandas as pd, numpy as np, shap
from sklearn.feature_extraction.text import TfidfVectorizer
from 00_config import DATA_PROCESSED, RESULTS, MODELS

OUT = RESULTS / "shap"
OUT.mkdir(parents=True, exist_ok=True)

def main(n_examples=200, top_k=10, seed=42):
    vec = joblib.load(MODELS / "vectorizer.joblib")
    clf = joblib.load(MODELS / "model.joblib")
    # SHAP needs dense 2D arrays; we’ll use a small background
    train = pd.read_csv(DATA_PROCESSED / "train.csv").sample(n=200, random_state=seed)
    test = pd.read_csv(DATA_PROCESSED / "test.csv").sample(n=min(n_examples, 200), random_state=seed)

    X_bg = vec.transform(train["text"]).toarray()
    X = vec.transform(test["text"]).toarray()

    explainer = shap.LinearExplainer(clf, X_bg, feature_perturbation="interventional")
    values = explainer.shap_values(X)  # shape: (n_examples, n_features)

    feature_names = vec.get_feature_names_out()
    records = []
    for i in range(X.shape[0]):
        scores = values[i]
        top_idx = np.argsort(np.abs(scores))[-top_k:][::-1]
        top = [(feature_names[j], float(scores[j])) for j in top_idx]
        records.append({"i": i, "text": test["text"].iloc[i], "pred": int(clf.predict(X[i:i+1])[0]), "top": top})
    pd.DataFrame(records).to_json(OUT / "shap_explanations.json", orient="records", lines=False)
    print("Saved SHAP explanations to", OUT)

if __name__ == "__main__":
    main()

