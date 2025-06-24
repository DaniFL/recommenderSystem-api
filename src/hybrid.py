# src/hybrid.py (Versión final y optimizada)

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

class HybridRecommender:
    def __init__(self, models_dir, data_dir):
        print("--- Inicializando Recomendador Híbrido Optimizado ---")
        
        # --- Carga de artefactos del modelo de contenido (sin cambios) ---
        content_model_path = Path(models_dir) / "clean_data"
        self.content_pipe = joblib.load(content_model_path / "steam_content_pipeline.joblib")
        self.content_features_svd = np.load(content_model_path / "steam_features_svd.npy")
        self.content_knn = joblib.load(content_model_path / "steam_knn_index.joblib")
        self.df_content = pd.read_csv(Path(data_dir) / "steam_games_clean.csv", usecols=['name'])
        self.game_name_to_idx = pd.Series(self.df_content.index, index=self.df_content['name'])
        
        # --- Carga de ARTEFACTOS LIGEROS del modelo colaborativo ---
        collab_model_path = Path(models_dir) / "collab_data_light" # Apuntamos a la nueva carpeta
        self.pu = np.load(collab_model_path / "user_factors.npy")
        self.qi = np.load(collab_model_path / "item_factors.npy")
        self.bu = np.load(collab_model_path / "user_biases.npy")
        self.bi = np.load(collab_model_path / "item_biases.npy")
        self.global_mean = joblib.load(collab_model_path / "global_mean.joblib")
        self.user_raw_to_inner = joblib.load(collab_model_path / "user_raw_to_inner.joblib")
        self.item_raw_to_inner = joblib.load(collab_model_path / "item_raw_to_inner.joblib")
        self.item_inner_to_raw = joblib.load(collab_model_path / "item_inner_to_raw.joblib")
        self.user_inner_to_rated = joblib.load(collab_model_path / "user_inner_to_rated.joblib")
        
        print("✅ Todos los modelos y datos optimizados han sido cargados.")

    def _predict_collab(self, user_inner_id, item_inner_id):
        """Reimplementación manual de la fórmula de predicción de SVD."""
        user_factor = self.pu[user_inner_id]
        item_factor = self.qi[item_inner_id]
        user_bias = self.bu[user_inner_id]
        item_bias = self.bi[item_inner_id]
        
        prediction = self.global_mean + user_bias + item_bias + np.dot(user_factor, item_factor)
        return prediction

    def get_collaborative_recommendations(self, user_id):
        """Genera recomendaciones colaborativas usando los artefactos ligeros."""
        user_inner_id = self.user_raw_to_inner.get(user_id)
        if user_inner_id is None:
            return {} # Usuario nuevo (cold start)

        rated_inner_items = set(self.user_inner_to_rated.get(user_inner_id, []))
        
        predictions = {}
        for item_raw_name, item_inner_id in self.item_raw_to_inner.items():
            if item_inner_id not in rated_inner_items:
                predictions[item_raw_name] = self._predict_collab(user_inner_id, item_inner_id)
        
        return predictions

    # Las funciones _normalize_scores, get_content_recommendations y recommend se mantienen IGUAL
    # No es necesario volver a copiarlas aquí, solo asegúrate de que sigan en el archivo.

    def _normalize_scores(self, scores_dict):
        if not scores_dict: return {}
        max_score = max(scores_dict.values()); min_score = min(scores_dict.values())
        if max_score == min_score: return {game: 1.0 for game in scores_dict}
        return {game: (score - min_score) / (max_score - min_score) for game, score in scores_dict.items()}

    def get_content_recommendations(self, positive_games, n=50):
        if not positive_games: return {}
        game_indices = [self.game_name_to_idx[name] for name in positive_games if name in self.game_name_to_idx]
        if not game_indices: return {}
        distances, indices = self.content_knn.kneighbors(self.content_features_svd[game_indices], n_neighbors=n)
        rec_scores = {}
        for i, game_idx in enumerate(game_indices):
            for j, neighbor_idx in enumerate(indices[i]):
                neighbor_game_name = self.df_content.loc[neighbor_idx, 'name']
                similarity = 1 - distances[i][j]
                if neighbor_game_name not in positive_games:
                    rec_scores[neighbor_game_name] = rec_scores.get(neighbor_game_name, 0) + similarity
        return rec_scores

    def recommend(self, user_id, positive_games, n=10, w_content=0.4, w_collab=0.6):
        print(f"\n--- Generando recomendaciones híbridas para el usuario: {user_id} ---")
        collab_recs = self.get_collaborative_recommendations(user_id)
        content_recs = self.get_content_recommendations(positive_games)
        if not collab_recs:
            print("Usuario nuevo detectado (Cold Start). Usando solo recomendaciones basadas en contenido.")
            w_content = 1.0
            w_collab = 0.0
        norm_collab_recs = self._normalize_scores(collab_recs)
        norm_content_recs = self._normalize_scores(content_recs)
        hybrid_scores = {}
        all_candidates = set(norm_collab_recs.keys()) | set(norm_content_recs.keys())
        for game in all_candidates:
            collab_score = norm_collab_recs.get(game, 0)
            content_score = norm_content_recs.get(game, 0)
            hybrid_scores[game] = (w_collab * collab_score) + (w_content * content_score)
        sorted_recs = sorted(hybrid_scores.items(), key=lambda item: item[1], reverse=True)
        return sorted_recs[:n]