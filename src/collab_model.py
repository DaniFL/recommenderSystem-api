# -*- coding: utf-8 -*-

# --------------------------------------------------------------------------
# --- SCRIPT PARA SISTEMA DE RECOMENDACIÓN CON FILTRO COLABORATIVO ---
# --------------------------------------------------------------------------
# Este script implementa un sistema de recomendación de videojuegos utilizando
# el enfoque de Filtro Colaborativo basado en modelos (Matrix Factorization)
# con el algoritmo SVD.
#
# LIBRERÍAS REQUERIDAS:
# - pandas: para la manipulación de datos.
# - scikit-surprise: para construir y evaluar modelos de recomendación.
#
# INSTALACIÓN:
# pip install pandas scikit-surprise
# --------------------------------------------------------------------------

# --- 0. Importación de Librerías ---
import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
import heapq # Usado para obtener los N mejores elementos de forma eficiente

# --- 1. Carga y Preprocesamiento de Datos ---
print("--- Paso 1: Cargando y preprocesando los datos ---")

try:
    # Carga el dataset desde el archivo CSV.
    # !! IMPORTANTE !!: Si tu archivo se llama diferente o está en otra carpeta,
    #                   actualiza la siguiente línea.
    # df = pd.read_csv('tu_dataset_real_con_992k_filas.csv')
    df = pd.read_csv(r"src\data\filtered_reviews.csv")
    
    print(f"Dataset cargado exitosamente. Forma: {df.shape}")
    print("Primeras 5 filas:")
    print(df.head())

except FileNotFoundError:
    print("Error: El archivo 'filtered_reviews.csv' no fue encontrado.")
    print("Asegúrate de que el archivo esté en el mismo directorio que el script.")
    exit()

# Creamos una columna 'rating' numérica basada en la recomendación.
# Este es nuestro feedback explícito.
#  - Recommended -> 1.0
#  - Not Recommended -> 0.0
df['rating'] = df['recommendation'].apply(lambda x: 1.0 if x == 'Recommended' else 0.0)

# Seleccionamos solo las columnas necesarias para el modelo de filtro colaborativo.
# El modelo necesita: un identificador de usuario, un identificador de ítem y un rating.
df_model = df[['user_id', 'game_name', 'rating']].copy()

# El 'Reader' le indica a la librería Surprise el rango de nuestros ratings.
# En nuestro caso, la escala es de 0 a 1.
reader = Reader(rating_scale=(0.0, 1.0))

# Cargamos el DataFrame en el formato de dataset que Surprise puede utilizar.
data = Dataset.load_from_df(df_model, reader)
print("\nDatos preprocesados y listos para el modelo.")


# --- 2. Entrenamiento del Modelo ---
print("\n--- Paso 2: Entrenando el modelo de recomendación (SVD) ---")

# Para hacer recomendaciones, entrenaremos el modelo con el conjunto de datos completo.
# Esto asegura que el modelo aprenda de todas las interacciones disponibles.
trainset = data.build_full_trainset()

# Instanciamos el algoritmo SVD (Singular Value Decomposition).
# - n_factors: El número de factores latentes (características ocultas).
# - n_epochs: El número de iteraciones del proceso de optimización.
# - random_state: Para asegurar reproducibilidad.
svd = SVD(n_factors=100, n_epochs=20, random_state=42)

# Entrenamos el modelo.
svd.fit(trainset)
print("Modelo entrenado exitosamente.")


# --- 3. Función para Generar Recomendaciones ---
print("\n--- Paso 3: Preparando la función para generar recomendaciones ---")

def get_top_n_recommendations(user_id, model, trainset, n=5):
    """
    Genera las N mejores recomendaciones para un usuario dado.
    
    Args:
        user_id (int/str): El ID del usuario para el que se generan las recomendaciones.
        model (surprise.prediction_algorithms.algo_base.AlgoBase): El modelo entrenado.
        trainset (surprise.Trainset): El conjunto de entrenamiento completo.
        n (int): El número de recomendaciones a generar.
        
    Returns:
        list: Una lista de tuplas (game_name, predicted_score).
    """
    # Obtener la lista de todos los juegos disponibles en el dataset.
    all_games = [trainset.to_raw_iid(inner_id) for inner_id in trainset.all_items()]
    
    # Verificar si el usuario está en el trainset.
    try:
        # Convertir el ID "raw" del usuario a un ID "interno" que usa Surprise.
        user_inner_id = trainset.to_inner_uid(user_id)
    except ValueError:
        print(f"Advertencia: El usuario '{user_id}' no está en el conjunto de datos. No se pueden generar recomendaciones personalizadas.")
        # Podríamos devolver los juegos más populares como fallback, pero por ahora devolvemos una lista vacía.
        return []
    
    # Obtener la lista de juegos que el usuario YA ha calificado.
    rated_games = {trainset.to_raw_iid(item_id) for (item_id, _) in trainset.ur[user_inner_id]}
    
    # Crear la lista de juegos que el usuario AÚN NO ha calificado.
    unrated_games = set(all_games) - rated_games
    
    # Predecir el rating para cada juego no calificado.
    predictions = []
    for game_name in unrated_games:
        predicted_rating = model.predict(uid=user_id, iid=game_name).est
        predictions.append((game_name, predicted_rating))
        
    # Ordenar las predicciones por la puntuación predicha (de mayor a menor)
    # y devolver las N mejores.
    top_n = heapq.nlargest(n, predictions, key=lambda x: x[1])
    
    return top_n

# --- Añadir esto al final de tu script de filtro colaborativo ---
import joblib
from pathlib import Path

# Definir rutas de salida
OUT_DIR_COLLAB = Path("src/models/collab_data") 
OUT_DIR_COLLAB.mkdir(parents=True, exist_ok=True)
COLLAB_MODEL_PATH = OUT_DIR_COLLAB / "collab_svd_model.joblib"
COLLAB_TRAINSET_PATH = OUT_DIR_COLLAB / "collab_trainset.joblib"

# Guardar el modelo SVD y el trainset
joblib.dump(svd, COLLAB_MODEL_PATH)
joblib.dump(trainset, COLLAB_TRAINSET_PATH)

print(f"\n✅ Modelo colaborativo y trainset guardados en {OUT_DIR_COLLAB.resolve()}")