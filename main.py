# main.py

import os
import sys
from pathlib import Path
from typing import List, Dict
from fastapi import FastAPI
from pydantic import BaseModel, Field

SRC_PATH = Path(__file__).parent / "src"
sys.path.append(str(SRC_PATH))

from hybrid import HybridRecommender

# --------------------------------------------------------------------------
# --- 1. CONFIGURACIÓN E INICIALIZACIÓN DE LA APP ---
# --------------------------------------------------------------------------

# Descripción para los metadatos de la API (se verá en /docs)
DESCRIPTION = """
API para un sistema de recomendación híbrido de videojuegos de Steam.
Permite obtener recomendaciones personalizadas combinando:
- **Filtro Basado en Contenido**: A partir de los juegos que le gustan a un usuario.
- **Filtro Colaborativo**: Basado en el comportamiento de usuarios similares.
"""

# Crea la instancia de la aplicación FastAPI
app = FastAPI(
    title="Steam Recommender API",
    description="API para recomendar videojuegos de Steam utilizando un sistema híbrido.",
    version="0.0.2"
)

# --------------------------------------------------------------------------
# --- 2. CARGA DEL MODELO ---
# --------------------------------------------------------------------------

# Obtiene las rutas desde variables de entorno para mayor flexibilidad en producción.
# Si no existen, usa las rutas locales por defecto.
MODELS_DIR = os.getenv("MODELS_DIR", "src/models")
DATA_DIR = os.getenv("DATA_DIR", "src/data")

# --- ¡PUNTO CLAVE! ---
# Instanciamos el recomendador UNA SOLA VEZ al iniciar la aplicación.
# Esto carga todos los modelos y datos en memoria al arrancar el servidor.
# De esta forma, las peticiones posteriores son súper rápidas, ya que no
# tienen que volver a cargar los pesados archivos .joblib y .npy.
print("Cargando modelos y datos en memoria, por favor espere...")
recommender = HybridRecommender(models_dir=MODELS_DIR, data_dir=DATA_DIR)
print("✅ Modelos y datos cargados exitosamente. La API está lista.")


# --------------------------------------------------------------------------
# --- 3. DEFINICIÓN DEL CUERPO DE LA PETICIÓN (REQUEST BODY) ---
# --------------------------------------------------------------------------

class RecommendationRequest(BaseModel):
    """Define la estructura de datos que la API espera recibir."""
    user_id: str = Field(..., example="76561197970982479", description="ID del usuario en Steam.")
    positive_games: List[str] = Field(..., example=["Counter-Strike", "Portal 2"], description="Lista de juegos que el usuario ha valorado positivamente.")
    n: int = Field(10, gt=0, le=50, description="Número de recomendaciones a devolver.")
    w_content: float = Field(0.4, ge=0.0, le=1.0, description="Peso para el score del modelo de contenido.")
    w_collab: float = Field(0.6, ge=0.0, le=1.0, description="Peso para el score del modelo colaborativo.")

# --------------------------------------------------------------------------
# --- 4. ENDPOINTS DE LA API ---
# --------------------------------------------------------------------------

@app.get("/", tags=["Status"])
async def get_status():
    """Endpoint raíz para verificar que la API está funcionando."""
    return {"status": "ok", "message": "Bienvenido a la API de Recomendación de Steam"}

@app.post("/recommend", tags=["Recommendations"], response_model=List[Dict[str, float]])
async def get_recommendations(request: RecommendationRequest):
    """
    Genera y devuelve una lista de recomendaciones de videojuegos.
    """
    recommendations = recommender.recommend(
        user_id=request.user_id,
        positive_games=request.positive_games,
        n=request.n,
        w_content=request.w_content,
        w_collab=request.w_collab
    )
    # Convierte la lista de tuplas [('game', score)] a una lista de dicts
    # que es un formato JSON más estándar: [{'game': score}]
    return [{game: score} for game, score in recommendations]