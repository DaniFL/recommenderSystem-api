# main.py
import os
import pandas as pd
import joblib
from surprise import dump
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

# =============================================================================
# CARGA DE MODELOS AL INICIO DE LA APLICACIÓN
# =============================================================================

MODELS = {}

# El decorador @app.on_event("startup") asegura que esta función se ejecute
# una sola vez cuando FastAPI se inicia. Es la forma correcta y más eficiente.
def cargar_modelos_y_datos():
    """
    Carga todos los modelos y datos en memoria una sola vez al inicio.
    Esto evita la carga repetitiva en cada petición a la API.
    """
    print("▶️  Iniciando la carga de modelos y datos...")

    rutas = {
        'preprocessor': 'model_files/content_preprocessor.joblib',
        'svd': 'model_files/content_svd_model.joblib',
        'knn_content': 'model_files/content_knn_model.joblib',
        'knn_collab': 'model_files/knn_collaborative_model.surprise',
        'games_df': 'model_files/steam_games_final.csv',
        'reviews_df': 'model_files/final_reviews_dataset.csv'
    }

    for nombre, ruta in rutas.items():
        if not os.path.exists(ruta):
            raise FileNotFoundError(f"Error crítico: No se encontró '{ruta}'. La API no puede iniciar.")

        if ruta.endswith('.surprise'):
            _, MODELS[nombre] = dump.load(ruta)
        elif ruta.endswith('.joblib'):
            MODELS[nombre] = joblib.load(ruta)
        elif ruta.endswith('.csv'):
            if nombre == 'reviews_df':
                # ----- OPTIMIZACIÓN DE MEMORIA -----
                # Para el archivo grande de reviews, cargamos solo las columnas necesarias.
                print(f"Cargando {ruta} (solo columnas esenciales para ahorrar memoria)...")
                MODELS[nombre] = pd.read_csv(
                    ruta,
                    usecols=['author_steamid', 'appid'], # ¡Este es el cambio clave!
                    encoding='latin-1',
                    engine='python',
                    on_bad_lines='skip'
                )
            else: # Para el archivo de juegos, que es más pequeño.
                MODELS[nombre] = pd.read_csv(
                    ruta,
                    dtype={14: str},
                    encoding='latin-1',
                    engine='python',
                    on_bad_lines='skip'
                )

    print("✅ Todos los artefactos cargados.")

    # --- Pre-cómputo de características y mapeos (sin cambios aquí) ---
    df_games = MODELS['games_df']
    df_games['required_age'] = pd.to_numeric(df_games['required_age'], errors='coerce').fillna(0)
    df_games['developers_main'] = df_games['developers'].str.split(',|;').str[0].str.strip().fillna('unknown')
    df_games['text_features'] = (df_games['name'].fillna('') + ' ' +
                                 df_games['short_description'].fillna('') + ' ' +
                                 df_games['genres'].fillna('').replace(';', ' ') + ' ' +
                                 df_games['categories'].fillna('').replace(';', ' '))

    X_processed = MODELS['preprocessor'].transform(df_games)
    MODELS['X_latent'] = MODELS['svd'].transform(X_processed)

    df_games.reset_index(drop=True, inplace=True)
    MODELS['name_to_idx'] = pd.Series(df_games.index, index=df_games['name']).drop_duplicates()
    MODELS['known_users'] = set(MODELS['reviews_df']['author_steamid'].unique())
    MODELS['all_games_appids'] = df_games['appid'].unique()

    print("🚀 ¡Aplicación lista para recibir peticiones!")


# =============================================================================
# DEFINICIÓN DE LA API CON FASTAPI
# =============================================================================

app = FastAPI(
    title="API de Recomendación de Juegos",
    description="Sistema híbrido para recomendar juegos de Steam."
)

# Esto ejecuta la carga de modelos al iniciar el servidor
@app.on_event("startup")
async def startup_event():
    cargar_modelos_y_datos()

# Modelo de datos para la petición (lo que el usuario nos enviará)
class RecommendationRequest(BaseModel):
    user_id: int
    game_titles: List[str]
    k: int = 10

# Endpoint principal de la API
@app.post("/recommend/")
async def obtener_recomendaciones_hibridas(request: RecommendationRequest):
    """
    Genera una lista de recomendaciones de juegos basada en un ID de usuario y UNA LISTA de títulos.
    """
    # --- a) Recomendaciones de Contenido (Lógica modificada) ---
    
    valid_indices = []
    invalid_titles = []
    
    # 1. Validar cada título de la lista y recoger sus índices
    for title in request.game_titles:
        if title in MODELS['name_to_idx']:
            valid_indices.append(MODELS['name_to_idx'][title])
        else:
            invalid_titles.append(title)

    # Si ninguno de los juegos proporcionados es válido, devuelve un error
    if not valid_indices:
        raise HTTPException(
            status_code=404,
            detail=f"Ninguno de los juegos proporcionados fue encontrado: {request.game_titles}"
        )

    # 2. Calcular el vector promedio de los juegos válidos
    latent_vectors = [MODELS['X_latent'][idx] for idx in valid_indices]
    query_vector = np.mean(latent_vectors, axis=0).reshape(1, -1)

    # 3. Encontrar vecinos al vector promedio
    # Pedimos más vecinos para poder filtrar los juegos de entrada
    num_neighbors = request.k + len(valid_indices)
    distances, indices = MODELS['knn_content'].kneighbors(query_vector, n_neighbors=num_neighbors)
    
    # 4. Filtrar los juegos de entrada de los resultados
    input_game_names = set(request.game_titles)
    content_recs_indices = indices.flatten()
    
    all_content_recs = MODELS['games_df']['name'].iloc[content_recs_indices]
    content_recs = [game for game in all_content_recs if game not in input_game_names][:request.k]

    # --- b) Lógica de Arranque en Frío (sin cambios) ---
    if request.user_id not in MODELS['known_users']:
        return {
            "type": "cold_start_user",
            "message": f"Usuario {request.user_id} no reconocido. Se devuelven recomendaciones basadas solo en contenido.",
            "recommendations": content_recs,
            "invalid_titles_skipped": invalid_titles if invalid_titles else None
        }

    # --- c) Recomendaciones Colaborativas (sin cambios) ---
    user_played_appids = set(MODELS['reviews_df'][MODELS['reviews_df']['author_steamid'] == request.user_id]['appid'])
    games_to_predict_appids = np.setdiff1d(list(MODELS['all_games_appids']), list(user_played_appids))

    predictions = [MODELS['knn_collab'].predict(request.user_id, game_id) for game_id in games_to_predict_appids]
    predictions.sort(key=lambda x: x.est, reverse=True)

    collab_recs_appids = [pred.iid for pred in predictions[:request.k]]
    collab_recs = list(MODELS['games_df'][MODELS['games_df']['appid'].isin(collab_recs_appids)]['name'])

    # --- d) Hibridación (sin cambios) ---
    hybrid_list = []
    content_idx, collab_idx = 0, 0
    while len(hybrid_list) < (request.k * 2):
        if content_idx < len(content_recs):
            hybrid_list.append(content_recs[content_idx])
            content_idx += 1
        if collab_idx < len(collab_recs):
            hybrid_list.append(collab_recs[collab_idx])
            collab_idx += 1
        if content_idx >= len(content_recs) and collab_idx >= len(collab_recs):
            break

    final_recs = list(pd.Series(hybrid_list).drop_duplicates().head(request.k))

    return {
        "type": "hybrid",
        "recommendations": final_recs,
        "invalid_titles_skipped": invalid_titles if invalid_titles else None
    }

@app.get("/")
def root():
    return {"message": "Bienvenido a la API de Recomendación. Visita /docs para interactuar."}