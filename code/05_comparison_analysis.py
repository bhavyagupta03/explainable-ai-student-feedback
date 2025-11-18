"""
Compare token-level overlap between LIME and SHAP; compute simple theme agreement.
Saves CSV + summary to figures/
"""
import json, pandas as pd, numpy as np
from 00_config import FIGURES, RESULTS

def load_json(path): 
    with open(path, "r", encoding="utf-8") as f: 
        return json.load(f)

def tokens_from_lime(weights_dict, top_k=10):
    # weights_dict is {token: weight}
    return [t for t,_ in sorted(weights_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:top_k]]

def tokens_from_shap(top_list):
    # top_list is [(token, score), ...]
    return [t for t,_ in top_list]

def jaccard(a, b):
    sa, sb = set(a), set(b)
    if not sa and not sb: return 1.0
    return len(sa & sb) / max(1, len(sa | sb))

def main(top_k=10):
    lime = load_json(RESULTS / "lime" / "lime_explanations.json")
    shap = load_json(RESULTS / "shap" / "shap_explanations.json")
    rows = []
    for i in range(min(len(lime), len(shap))):
        ltoks = tokens_from_lime(lime[i]["weights"], top_k=top_k)
        stoks = tokens_from_shap(shap[i]["top"])
        rows.append({"i": i, "overlap_jaccard": jaccard(ltoks, stoks)})
    df = pd.DataFrame(rows)
    df.to_csv(FIGURES / "token_overlap.csv", index=False)
    print(df.describe())

if __name__ == "__main__":
    main()

