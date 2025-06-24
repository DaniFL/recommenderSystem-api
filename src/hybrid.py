# filepath: recommendation-api/recommendation-api/src/hybrid.py
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from utils import MultiHotBinarizer

class HybridRecommender:
    def __init__(self, models_dir, data_dir):
        print("--- Inicializando Recomendador Híbrido ---")
        
        content_model_path = Path(models_dir) / "clean_data"
        self.content_pipe = joblib.load(content_model_path / "steam_content_pipeline.joblib")
        self.content_features_svd = np.load(content_model_path / "steam_features_svd.npy")
        self.content_knn = joblib.load(content_model_path / "steam_knn_index.joblib")
        self.df_content = pd.read_csv(Path(data_dir) / "steam_games_clean.csv")
        self.game_name_to_idx = pd.Series(self.df_content.index, index=self.df_content['name'])
        
        collab_model_path = Path(models_dir) / "collab_data"
        self.collab_model = joblib.load(collab_model_path / "collab_svd_model.joblib")
        self.collab_trainset = joblib.load(collab_model_path / "collab_trainset.joblib")
        
        print("✅ Todos los modelos y datos han sido cargados.")

    def _normalize_scores(self, scores_dict):
        if not scores_dict:
            return {}
        max_score = max(scores_dict.values()) if scores_dict else 0
        min_score = min(scores_dict.values()) if scores_dict else 0
        if max_score == min_score:
            return {game: 1.0 for game in scores_dict}
        return {game: (score - min_score) / (max_score - min_score) for game, score in scores_dict.items()}

    def get_content_recommendations(self, positive_games, n=50):
        if not positive_games:
            return {}
        game_indices = [self.game_name_to_idx[name] for name in positive_games if name in self.game_name_to_idx]
        if not game_indices:
            return {}
        distances, indices = self.content_knn.kneighbors(self.content_features_svd[game_indices], n_neighbors=n)
        rec_scores = {}
        for i, game_idx in enumerate(game_indices):
            for j, neighbor_idx in enumerate(indices[i]):
                neighbor_game_name = self.df_content.loc[neighbor_idx, 'name']
                similarity = 1 - distances[i][j]
                if neighbor_game_name not in positive_games:
                    rec_scores[neighbor_game_name] = rec_scores.get(neighbor_game_name, 0) + similarity
        return rec_scores

    def get_collaborative_recommendations(self, user_id):
        try:
            user_inner_id = self.collab_trainset.to_inner_uid(user_id)
        except ValueError:
            return {}
        all_game_names = [self.collab_trainset.to_raw_iid(inner_id) for inner_id in self.collab_trainset.all_items()]
        rated_games = {self.collab_trainset.to_raw_iid(item_id) for (item_id, _) in self.collab_trainset.ur[user_inner_id]}
        unrated_games = set(all_game_names) - rated_games
        predictions = {game: self.collab_model.predict(user_id, game).est for game in unrated_games}
        return predictions

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
            hybrid_score = (w_collab * collab_score) + (w_content * content_score)
            hybrid_scores[game] = hybrid_score
        sorted_recs = sorted(hybrid_scores.items(), key=lambda item: item[1], reverse=True)
        return sorted_recs[:n]