# utils.py
import pandas as pd
import ast
import re
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MultiLabelBinarizer

class MultiHotBinarizer(BaseEstimator, TransformerMixin):
    def __init__(self, min_freq=5):
        self.min_freq = min_freq
        self.mlb = MultiLabelBinarizer()
    def fit(self, X, y=None):
        # filtra etiquetas poco frecuentes
        counts = {}
        for row in X:
            for lab in row: counts[lab] = counts.get(lab,0)+1
        self.valid_labels_ = {l for l,c in counts.items() if c>=self.min_freq}
        X_filtered = [[l for l in row if l in self.valid_labels_] for row in X]
        self.mlb.fit(X_filtered)
        return self
    def transform(self, X):
        X_filtered = [[l for l in row if l in self.valid_labels_] for row in X]
        return self.mlb.transform(X_filtered)

def parse_list_field(x):
    if pd.isna(x): return []
    try:
        val = ast.literal_eval(x)
        return [v.strip().lower() for v in val] if isinstance(val, list) else []
    except (ValueError, SyntaxError):
        return [v.strip().lower() for v in re.split(r"[;,]", str(x)) if v]
