#!/usr/bin/env python
# build_steam_content_model.py
# ───────────────────────────────────────────────────────────
# Incluye developers, languages, required_age + texto/genres.

import joblib, nltk, numpy as np, pandas as pd
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors

# ✅ ¡CORRECCIÓN! Importamos todo desde nuestro módulo de utilidades.
from utils import MultiHotBinarizer, parse_list_field

# ── Rutas ──────────────────────────────────────────────────
CSV_IN   = Path(r"src\data\steam_games_clean.csv")
OUT_DIR  = Path(r"src\models\clean_data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PIPE_PATH = OUT_DIR/"steam_content_pipeline.joblib"
MAT_PATH  = OUT_DIR/"steam_features_svd.npy"
KNN_PATH  = OUT_DIR/"steam_knn_index.joblib"

# ── 1. Leer dataset ────────────────────────────────────────
df = pd.read_csv(CSV_IN)

# ── 2. Columnas custom (listas) ────────────────────────────
# ✅ ¡CORRECCIÓN! Usamos la función importada.
df["dev_main"]   = df["developers"].str.split(";").str[0].str.strip().fillna("unknown")
df["langs"]      = df["supported_languages"].apply(parse_list_field)
df["genres_lst"] = df["genres"].apply(parse_list_field)

df["text"] = (df["name"].fillna("")   + " " +
              df["short_description"].fillna("")).str.lower()

# ❌ ¡CORRECCIÓN! Hemos ELIMINADO las definiciones locales de 'parse_list_field'
#    y 'MultiHotBinarizer' de esta sección, ya que ahora se importan.

# ── 4. Transformadores ────────────────────────────────────
nltk.download('stopwords', quiet=True) # Asegura que los stopwords estén descargados
stop_es = nltk.corpus.stopwords.words("spanish")
stop_en = nltk.corpus.stopwords.words("english")
tfidf = TfidfVectorizer(max_features=30_000,
                        stop_words=stop_es+stop_en,
                        ngram_range=(1,2),
                        lowercase=True)

ohe_dev  = OneHotEncoder(handle_unknown="ignore", min_frequency=10)
bin_lang = MultiHotBinarizer(min_freq=20)
bin_gen  = MultiHotBinarizer(min_freq=50)
sc_age   = StandardScaler()

pre = ColumnTransformer([
        ("txt",  tfidf,    "text"),
        ("dev",  ohe_dev,  ["dev_main"]),
        ("lang", bin_lang, "langs"),
        ("gen",  bin_gen,  "genres_lst"),
        ("age",  sc_age,   ["required_age"])
    ], sparse_threshold=0.7, remainder="drop", n_jobs=-1)

svd = TruncatedSVD(n_components=300, random_state=42)
pipe = Pipeline([("pre", pre), ("svd", svd)])

print("▶️  Ajustando pipeline…")
X_latent = pipe.fit_transform(df)
print("Matriz latente:", X_latent.shape)

# ── 5. Índice k-NN ────────────────────────────────────────
knn = NearestNeighbors(metric="cosine", algorithm="brute")
knn.fit(X_latent)

# ── 6. Guardar artefactos ─────────────────────────────────
joblib.dump(pipe, PIPE_PATH)
np.save(MAT_PATH, X_latent)
joblib.dump(knn, KNN_PATH)

print("✅  Guardados en", OUT_DIR.resolve())