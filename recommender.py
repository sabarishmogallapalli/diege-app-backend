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

    def recommend(self, user_prompt):
        """Translates user text to math, compares it to the baked brain, and returns matches."""
        # Encode strictly the single user sentence into a mathematical vector
        user_vector = self.encoder.encode([user_prompt])
        
        # Search the pre-baked matrix for the 6 closest geometric matches
        distances, indices = self.knn.kneighbors(user_vector)
        
        results = []
        for i, idx in enumerate(indices[0]):
            movie = self.df.iloc[idx]
            match_score = 1 - distances[0][i]
            
            results.append({
                "id": int(movie['id']),
                "title": movie['title'],
                "overview": movie['overview'],
                "vibe_match": float(match_score)
            })
            
        return results

# Initialize the global instance so main.py can trigger it
recommender = MovieRecommender()