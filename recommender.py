import os
# 1. Throttle PyTorch to prevent RAM spikes on the Render Free Tier
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import torch
torch.set_num_threads(1)

from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

class MovieRecommender:
    def __init__(self):
        print("Diege is waking up...")
        # Load the lightweight model ONLY for reading the user's live text prompt
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')

    def load_data(self, csv_path):
        print("Loading movie metadata...")
        # Load the movie dataset strictly to retrieve titles and IDs
        self.df = pd.read_csv(csv_path)
        self.df = self.df[['id', 'title', 'overview', 'genres', 'keywords', 'tagline']].fillna('')

        # 2. THE SILVER BULLET: Load the pre-calculated math instantly
        print("Loading baked brain matrix...")
        self.embeddings = np.load('movie_embeddings.npy')

        # Spin up the nearest neighbors matcher
        self.knn = NearestNeighbors(n_neighbors=6, metric='cosine')
        self.knn.fit(self.embeddings)
        print("Brain is ready.")

    def _clean_negations(self, prompt: str) -> str:
        """Strips out words immediately following common negation words."""
        negation_patterns = [
            r"(?:don't want|do not want|no|without|skip|instead of)\s+([\w\s]+?)(?=(?:,|\.|$|and))"
        ]
        cleaned = prompt.lower()
        for pattern in negation_patterns:
            cleaned = re.sub(pattern, "", cleaned)
        return cleaned.strip()
    
    def _apply_vibe_weights(self, prompt: str) -> str:
        """Duplicated key aesthetic words to visually amplify their vector math importance."""
        words = prompt.lower().split()
        weighted_tokens = []
        for word in words:
            clean_word = word.strip(",.!?")
            if clean_word in self.BOUTIQUE_BOOSTS:
                multiplier = self.BOUTIQUE_BOOSTS[clean_word]
                weighted_tokens.extend([word] * multiplier)
            else:
                weighted_tokens.append(word)
        return " ".join(weighted_tokens)

    def recommend(self, user_prompt, provider_filters=None):
        """Translates user text to math, compares it to the baked brain, and returns matches."""
        # Encode strictly the single user sentence into a mathematical vector
        cleaned_prompt = self._clean_negations(user_prompt)
        weighted_prompt = self._apply_vibe_weights(cleaned_prompt)
        user_vector = self.encoder.encode([user_prompt])
        
        # Search the pre-baked matrix for the 6 closest geometric matches
        #distances, indices = self.knn.kneighbors(user_vector)
        
        if provider_filters:
            provider_filters = [int(p) for p in provider_filters]
            matched_indices = [
                idx for idx, movie in self.df.iterrows() 
                if any(p in movie.get('providers', []) for p in provider_filters)
            ]
            # filtered_embeddings = self.embeddings[matched_indices]
            # local_knn = NearestNeighbors(n_neighbors=min(6, len(matched_indices)), metric='cosine')
            # local_knn.fit(filtered_embeddings)
            # distances, local_indices = local_knn.kneighbors(user_vector)
            # indices = [[matched_indices[x] for x in local_indices[0]]]
        else:
            matched_indices = []

        results = []
        if provider_filters and matched_indices:
            filtered_embeddings = self.embeddings[matched_indices]
            local_knn = NearestNeighbors(n_neighbors=min(6, len(matched_indices)), metric='cosine')
            local_knn.fit(filtered_embeddings)
            distances, local_indices = local_knn.kneighbors(user_vector)

            for i, local_idx in enumerate(local_indices[0]):
                original_df_idx = matched_indices[local_idx]
                movie = self.df.iloc[original_df_idx]
                match_score = 1 - distances[0][i]
                results.append({
                    "id": int(movie['id']),
                    "title": movie['title'],
                    "overview": movie['overview'],
                    "vibe_match": float(match_score)
                })
        else:
            distances, indices = self.knn.kneighbors(user_vector)
            for i, idx in enumerate(indices[0]):
                movie = self.df.iloc[idx]
                match_score = 1 - distances[0][i]
                
                results.append({
                    "id": int(movie['id']),
                    "title": movie['title'],
                    "overview": movie['overview'],
                    "vibe_match": float(match_score)
                })
        is_low_confidence = len(results) > 0 and results[0]["vibe_match"] < 0.16
            
        return {
            "movies": results,
            "low_confidence": is_low_confidence
        }

# Initialize the global instance so main.py can trigger it
recommender = MovieRecommender()