# src/hybrid.py

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

class HybridRecommender:
    def __init__(self, models_dir, data_dir):
        """
        Inicializa el recomendador híbrido cargando todos los modelos
        y datos necesarios en memoria.
        """
        print("--- Inicializando Recomendador Híbrido ---")
        
        # --- Carga de artefactos del modelo de contenido ---
        content_model_path = Path(models_dir) / "clean_data"
        self.content_pipe = joblib.load(content_model_path / "steam_content_pipeline.joblib")
        self.content_features_svd = np.load(content_model_path / "steam_features_svd.npy")
        self.content_knn = joblib.load(content_model_path / "steam_knn_index.joblib")
        
        # Carga los datos de los juegos y crea un mapeo de nombre -> índice para búsquedas rápidas
        self.df_content = pd.read_csv(Path(data_dir) / "steam_games_clean.csv")
        self.game_name_to_idx = pd.Series(self.df_content.index, index=self.df_content['name'])
        
        # --- Carga de artefactos del modelo colaborativo ---
        collab_model_path = Path(models_dir) / "collab_data"
        self.collab_model = joblib.load(collab_model_path / "collab_svd_model.joblib")
        self.collab_trainset = joblib.load(collab_model_path / "collab_trainset.joblib")
        
        print("✅ Todos los modelos y datos han sido cargados.")

    def _normalize_scores(self, scores_dict):
        """Normaliza un diccionario de puntuaciones a una escala de 0 a 1."""
        if not scores_dict:
            return {}
        
        max_score = max(scores_dict.values())
        min_score = min(scores_dict.values())
        
        if max_score == min_score:
            return {game: 1.0 for game in scores_dict}
            
        return {game: (score - min_score) / (max_score - min_score) for game, score in scores_dict.items()}

    def get_content_recommendations(self, positive_games, n=50):
        """Genera recomendaciones basadas en contenido (similitud de juegos)."""
        if not positive_games:
            return {}
        
        # Obtener los índices de los juegos que le gustan al usuario
        game_indices = [self.game_name_to_idx[name] for name in positive_games if name in self.game_name_to_idx]
        
        if not game_indices:
            return {}
            
        # Encontrar los N vecinos más cercanos para los juegos de entrada
        distances, indices = self.content_knn.kneighbors(self.content_features_svd[game_indices], n_neighbors=n)
        
        # Calcular puntuaciones de recomendación basadas en la similitud
        rec_scores = {}
        for i, game_idx in enumerate(game_indices):
            for j, neighbor_idx in enumerate(indices[i]):
                neighbor_game_name = self.df_content.loc[neighbor_idx, 'name']
                similarity = 1 - distances[i][j]  # Convertir distancia coseno a similitud
                
                if neighbor_game_name not in positive_games:
                    rec_scores[neighbor_game_name] = rec_scores.get(neighbor_game_name, 0) + similarity
                    
        return rec_scores

    def get_collaborative_recommendations(self, user_id):
        """Genera recomendaciones basadas en filtro colaborativo (comportamiento de usuarios)."""
        try:
            # Comprobar si el usuario existe en el conjunto de entrenamiento
            user_inner_id = self.collab_trainset.to_inner_uid(user_id)
        except ValueError:
            # El usuario es nuevo, no se pueden generar recomendaciones colaborativas (cold start)
            return {}
            
        # Predecir puntuaciones para los juegos que el usuario no ha calificado
        all_game_names = [self.collab_trainset.to_raw_iid(inner_id) for inner_id in self.collab_trainset.all_items()]
        rated_games = {self.collab_trainset.to_raw_iid(item_id) for (item_id, _) in self.collab_trainset.ur[user_inner_id]}
        unrated_games = set(all_game_names) - rated_games
        
        predictions = {game: self.collab_model.predict(user_id, game).est for game in unrated_games}
        return predictions

    def recommend(self, user_id, positive_games, n=10, w_content=0.4, w_collab=0.6):
        """
        Genera la recomendación híbrida final combinando ambos enfoques.
        """
        print(f"\n--- Generando recomendaciones híbridas para el usuario: {user_id} ---")
        
        # Obtener recomendaciones de ambos modelos
        collab_recs = self.get_collaborative_recommendations(user_id)
        content_recs = self.get_content_recommendations(positive_games)
        
        # Manejo del caso de "Cold Start": si no hay recs colaborativas, usar solo contenido.
        if not collab_recs:
            print("Usuario nuevo detectado (Cold Start). Usando solo recomendaciones basadas en contenido.")
            w_content = 1.0
            w_collab = 0.0
        
        # Normalizar las puntuaciones de ambos modelos a una escala [0, 1]
        norm_collab_recs = self._normalize_scores(collab_recs)
        norm_content_recs = self._normalize_scores(content_recs)
        
        # Combinar las puntuaciones de forma ponderada
        hybrid_scores = {}
        all_candidates = set(norm_collab_recs.keys()) | set(norm_content_recs.keys())
        
        for game in all_candidates:
            collab_score = norm_collab_recs.get(game, 0)
            content_score = norm_content_recs.get(game, 0)
            hybrid_score = (w_collab * collab_score) + (w_content * content_score)
            hybrid_scores[game] = hybrid_score
            
        # Ordenar y devolver las N mejores recomendaciones
        sorted_recs = sorted(hybrid_scores.items(), key=lambda item: item[1], reverse=True)
        return sorted_recs[:n]