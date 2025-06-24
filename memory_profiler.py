# memory_profiler.py (Versión final y corregida para artefactos ligeros)

import os
import sys
import psutil
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# --- Añade el directorio 'src' al path de Python ---
SRC_PATH = Path(__file__).parent / "src"
sys.path.append(str(SRC_PATH))

# --- Función para medir y mostrar la memoria ---
def print_memory_usage(step_name=""):
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    mem_mb = mem_info.rss / (1024 * 1024) # rss: Resident Set Size
    print(f"[{step_name}] Uso de memoria actual: {mem_mb:.2f} MB")

# --- Rutas a los artefactos ---
MODELS_DIR = Path("src/models")
DATA_DIR = Path("src/data")

# --- Inicio del perfilado ---
print("--- Iniciando diagnóstico de memoria ---")
print_memory_usage("Inicio")

# --- Cargando artefactos de Contenido ---
print("\n--- Modelos de Contenido ---")
try:
    content_model_path = MODELS_DIR / "clean_data"
    
    obj = joblib.load(content_model_path / "steam_content_pipeline.joblib")
    print_memory_usage("Carga de 'content_pipeline.joblib'")
    del obj
    
    obj = pd.read_csv(DATA_DIR / "steam_games_clean.csv", usecols=['name'])
    print_memory_usage("Carga de 'steam_games_clean.csv' (solo name)")
    del obj

    obj = np.load(MODELS_DIR / "clean_data" / "steam_features_svd.npy")
    print_memory_usage("Carga de 'features_svd.npy'")
    del obj
except Exception as e:
    print(f"Error cargando un artefacto de contenido: {e}")

# --- Cargando artefactos Colaborativos LIGEROS ---
print("\n--- Modelos Colaborativos (Ligeros) ---")
try:
    collab_model_path = MODELS_DIR / "collab_data_light" # Apuntamos a la nueva carpeta
    
    obj = np.load(collab_model_path / "user_factors.npy")
    print_memory_usage("Carga de 'user_factors.npy'")
    del obj
    
    obj = np.load(collab_model_path / "item_factors.npy")
    print_memory_usage("Carga de 'item_factors.npy'")
    del obj
    
    obj = np.load(collab_model_path / "user_biases.npy")
    print_memory_usage("Carga de 'user_biases.npy'")
    del obj

    obj = np.load(collab_model_path / "item_biases.npy")
    print_memory_usage("Carga de 'item_biases.npy'")
    del obj

    obj = joblib.load(collab_model_path / "global_mean.joblib")
    print_memory_usage("Carga de 'global_mean.joblib'")
    del obj
    
    obj = joblib.load(collab_model_path / "user_raw_to_inner.joblib")
    print_memory_usage("Carga de 'user_raw_to_inner.joblib'")
    del obj

    obj = joblib.load(collab_model_path / "item_raw_to_inner.joblib")
    print_memory_usage("Carga de 'item_raw_to_inner.joblib'")
    del obj

    obj = joblib.load(collab_model_path / "item_inner_to_raw.joblib")
    print_memory_usage("Carga de 'item_inner_to_raw.joblib'")
    del obj

    obj = joblib.load(collab_model_path / "user_inner_to_rated.joblib")
    print_memory_usage("Carga de 'user_inner_to_rated.joblib'")
    del obj

except Exception as e:
    print(f"Error cargando un artefacto colaborativo: {e}")

print("\n--- Diagnóstico de memoria finalizado ---")