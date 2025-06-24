# collab_model.py (Versión final y optimizada)

import pandas as pd
from surprise import Dataset, Reader, SVD
import joblib
from pathlib import Path
import numpy as np

print("--- Paso 1: Cargando y preprocesando los datos ---")

try:
    df = pd.read_csv(r"src/data/filtered_reviews.csv")
    print(f"Dataset cargado exitosamente. Forma: {df.shape}")
except FileNotFoundError:
    print("Error: El archivo 'filtered_reviews.csv' no fue encontrado.")
    exit()

df['rating'] = df['recommendation'].apply(lambda x: 1.0 if x == 'Recommended' else 0.0)
df_model = df[['user_id', 'game_name', 'rating']].copy()
reader = Reader(rating_scale=(0.0, 1.0))
data = Dataset.load_from_df(df_model, reader)

print("\n--- Paso 2: Entrenando el modelo de recomendación (SVD) ---")
trainset = data.build_full_trainset()
svd = SVD(n_factors=100, n_epochs=20, random_state=42, verbose=True)
svd.fit(trainset)
print("Modelo entrenado exitosamente.")

# --- Paso 3: EXTRACCIÓN Y GUARDADO DE ARTEFACTOS LIGEROS ---
print("\n--- Paso 3: Extrayendo y guardando artefactos ligeros ---")

OUT_DIR_COLLAB = Path("src/models/collab_data_light") # Usamos una nueva carpeta
OUT_DIR_COLLAB.mkdir(parents=True, exist_ok=True)

# 1. Extraer y guardar los componentes del modelo SVD
# Estos son arrays de numpy, muy eficientes en memoria.
np.save(OUT_DIR_COLLAB / "user_factors.npy", svd.pu)
np.save(OUT_DIR_COLLAB / "item_factors.npy", svd.qi)
np.save(OUT_DIR_COLLAB / "user_biases.npy", svd.bu)
np.save(OUT_DIR_COLLAB / "item_biases.npy", svd.bi)
joblib.dump(trainset.global_mean, OUT_DIR_COLLAB / "global_mean.joblib")

# 2. Extraer y guardar los mapeos del trainset
# Estos son diccionarios, mucho más ligeros que el objeto trainset completo.
user_raw_to_inner = {raw_id: inner_id for raw_id, inner_id in trainset._raw2inner_id_users.items()}
item_raw_to_inner = {raw_id: inner_id for raw_id, inner_id in trainset._raw2inner_id_items.items()}
item_inner_to_raw = {inner_id: raw_id for raw_id, inner_id in item_raw_to_inner.items()}
user_inner_to_rated = {inner_id: [item for item, _ in ratings] for inner_id, ratings in trainset.ur.items()}

joblib.dump(user_raw_to_inner, OUT_DIR_COLLAB / "user_raw_to_inner.joblib")
joblib.dump(item_raw_to_inner, OUT_DIR_COLLAB / "item_raw_to_inner.joblib")
joblib.dump(item_inner_to_raw, OUT_DIR_COLLAB / "item_inner_to_raw.joblib")
joblib.dump(user_inner_to_rated, OUT_DIR_COLLAB / "user_inner_to_rated.joblib")

print(f"\n✅ Todos los artefactos colaborativos ligeros han sido guardados en {OUT_DIR_COLLAB.resolve()}")