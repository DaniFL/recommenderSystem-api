# main.py
import os
import pandas as pd
import joblib
from surprise import dump
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import re

# =============================================================================
# FUNCIÓN DE NORMALIZACIÓN
# =============================================================================

def normalize_title(title: str) -> str:
    """
    Limpia un título de juego para una comparación robusta:
    - Elimina caracteres especiales (como ®, ™, ©, etc.).
    - Elimina espacios en blanco al inicio y al final.
    """
    if not isinstance(title, str):
        return ""
    # Elimina cualquier caracter que no sea una letra, número, espacio o guion
    normalized = re.sub(r'[^\w\s-]', '', title)
    return normalized.strip()

# =============================================================================
# CARGA DE MODELOS AL INICIO DE LA APLICACIÓN
# =============================================================================

MODELS = {}

def cargar_modelos_y_datos():
    """
    Carga todos los modelos y datos en memoria una sola vez al inicio.
    """
    print("Iniciando la carga de modelos y datos...")

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
            # (El código de carga de CSV no cambia)
            if nombre == 'reviews_df':
                MODELS[nombre] = pd.read_csv(ruta, usecols=['author_steamid', 'appid'], encoding='latin-1', engine='python', on_bad_lines='skip')
            else:
                MODELS[nombre] = pd.read_csv(ruta, dtype={14: str}, encoding='latin-1', engine='python', on_bad_lines='skip')

    print("Todos los artefactos cargados.")

    df_games = MODELS['games_df']
    
    # Rellenamos nulos para evitar errores posteriores
    text_cols = ['name', 'short_description', 'genres', 'categories', 'developers']
    for col in text_cols:
        df_games[col] = df_games[col].fillna('')

    
    df_games['name_normalized'] = df_games['name'].apply(normalize_title)
    
    # La ingeniería de características se hace con los datos originales para coincidir con el entrenamiento
    df_games['required_age'] = pd.to_numeric(df_games['required_age'], errors='coerce').fillna(0)
    df_games['developers_main'] = df_games['developers'].str.split(',|;').str[0].str.strip()
    df_games['text_features'] = (df_games['name'] + ' ' + df_games['short_description'] + ' ' +
                                 df_games['genres'].replace(';', ' ') + ' ' + df_games['categories'].replace(';', ' '))

    X_processed = MODELS['preprocessor'].transform(df_games)
    MODELS['X_latent'] = MODELS['svd'].transform(X_processed)

    df_games.reset_index(drop=True, inplace=True)
    
    MODELS['name_to_idx'] = pd.Series(df_games.index, index=df_games['name_normalized']).drop_duplicates()
    
    MODELS['known_users'] = set(MODELS['reviews_df']['author_steamid'].unique())
    MODELS['all_games_appids'] = df_games['appid'].unique()

    print("🚀 ¡Aplicación lista para recibir peticiones!")


# =============================================================================
# DEFINICIÓN DE LA API CON FASTAPI
# =============================================================================

app = FastAPI(title="API de Recomendación de Juegos", description="Sistema híbrido para recomendar juegos de Steam.")

@app.on_event("startup")
async def startup_event():
    cargar_modelos_y_datos()

class RecommendationRequest(BaseModel):
    user_id: int
    game_titles: List[str]
    k: int = 10

@app.post("/recommend/")
async def obtener_recomendaciones_hibridas(request: RecommendationRequest):
    valid_indices = []
    invalid_titles = []
    
    for title in request.game_titles:
        
        clean_title = normalize_title(title)
        
        if clean_title in MODELS['name_to_idx']:
            valid_indices.append(MODELS['name_to_idx'][clean_title])
        else:
            invalid_titles.append(title)

    if not valid_indices:
        raise HTTPException(status_code=404, detail=f"Ninguno de los juegos proporcionados fue encontrado: {request.game_titles}")

    latent_vectors = [MODELS['X_latent'][idx] for idx in valid_indices]
    query_vector = np.mean(latent_vectors, axis=0).reshape(1, -1)

    num_neighbors = request.k + len(valid_indices)
    distances, indices = MODELS['knn_content'].kneighbors(query_vector, n_neighbors=num_neighbors)
    
    input_game_names_normalized = {normalize_title(title) for title in request.game_titles}
    content_recs_indices = indices.flatten()
    
    # Se filtra usando los nombres normalizados, pero se devuelven los originales.
    df_games_ref = MODELS['games_df']
    filtered_indices = [
        idx for idx in content_recs_indices 
        if df_games_ref.loc[idx, 'name_normalized'] not in input_game_names_normalized
    ]
    
    content_recs = list(df_games_ref.loc[filtered_indices, 'name'].head(request.k))

    if request.user_id not in MODELS['known_users']:
        return {"type": "cold_start_user", "recommendations": content_recs, "invalid_titles_skipped": invalid_titles or None}

    # (La lógica del filtro colaborativo y la hibridación no cambia)
    user_played_appids = set(MODELS['reviews_df'][MODELS['reviews_df']['author_steamid'] == request.user_id]['appid'])
    games_to_predict_appids = np.setdiff1d(list(MODELS['all_games_appids']), list(user_played_appids))
    predictions = [MODELS['knn_collab'].predict(request.user_id, game_id) for game_id in games_to_predict_appids]
    predictions.sort(key=lambda x: x.est, reverse=True)
    collab_recs_appids = [pred.iid for pred in predictions[:request.k]]
    collab_recs = list(df_games_ref[df_games_ref['appid'].isin(collab_recs_appids)]['name'])

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

    return {"type": "hybrid", "recommendations": final_recs, "invalid_titles_skipped": invalid_titles or None}

@app.get("/")
def root():
    return {"message": "Bienvenido a la API de Recomendación. Visita /docs para interactuar."}