"""
Inspect misclassifications and save a simple report with predicted vs true labels.
"""
import pandas as pd, joblib
from 00_config import DATA_PROCESSED, MODELS, FIGURES

def main():
    vec = joblib.load(MODELS / "vectorizer.joblib")
    clf = joblib.load(MODELS / "model.joblib")
    test = pd.read_csv(DATA_PROCESSED / "test.csv")
    X = vec.transform(test["text"])
    pred = clf.predict(X)
    test["pred"] = pred
    errs = test[test["pred"] != test["label"]]
    errs[["text","label","pred"]].to_csv(FIGURES / "errors.csv", index=False)
    print(f"Saved {len(errs)} error examples to figures/errors.csv")

if __name__ == "__main__":
    main()

