import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import torch
torch.set_num_threads(1)

from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import ast
from sklearn.metrics.pairwise import cosine_similarity

class MovieRecommender:

    def __init__(self):
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.df = pd.read_csv('TMDB 5000 Movies.csv')
        #parameters I declare
        self.tmdb = None
        self.embeddings = np.load('movie_embeddings.npy')
        self.knn = NearestNeighbors(n_neighbors=6, metric='cosine')
        self.knn.fit(self.embeddings)
        print("Brain is ready.")
    
    # def load_data(self, csv_path):
    #     self.df = pd.read_csv(csv_path)
    #     self.df = self.df[['id', 'title', 'overview', 'genres', 'keywords', 'tagline']].fillna('')

    #     self.df['genres'] = self.df['genres'].apply(clean_json_list)
    #     self.df['keywords'] = self.df['keywords'].apply(clean_json_list)

    #     #THE ULTIMATE DATA ANALYSIS PROMPT
    #     self.df['vibe_features'] = (
    #         self.df['overview'] + " " + 
    #         self.df['genres'] + " " + 
    #         self.df['keywords'] + " " *2 + 
    #         self.df['tagline'] *2
    #     ).str.lower()

    #     #print("encoding the Movie Vibes")
    #     #self.embeddings = self.encoder.encode(self.df['vibe_features'].tolist(), show_progress_bar=True)
    #     self.knn = NearestNeighbors(n_neighbors=6, metric='cosine')
    #     self.knn.fit(self.embeddings)
    #     self.df['genres'] = self.df['genres'].apply(clean_json_list)
    #     self.df['keywords'] = self.df['keywords'].apply(clean_json_list)
    #     self.df['vibe_features'] = self.df['overview'] + " " + self.df['genres'] + " " + self.df['keywords']

    #     print("Diege Brain is ready.")

    def recommend(self, user_prompt):
        """Translates user text to math, compares it to the baked brain, and returns matches."""
        
        # Encode strictly the single user sentence into a mathematical vector
        user_vector = self.encoder.encode([user_prompt])
        
        # Search the pre-baked matrix for the 6 closest geometric matches
        distances, indices = self.knn.kneighbors(user_vector)
        
        results = []
        # Convert the raw index integers back into actual movie data
        for i, idx in enumerate(indices[0]):
            movie = self.df.iloc[idx]
            
            # Translate the cosine distance into a human-readable percentage score
            match_score = 1 - distances[0][i]
            
            results.append({
                "id": int(movie['id']),
                "title": movie['title'],
                "overview": movie['overview'],
                "vibe_match": float(match_score)
            })
            
        return results
recommender = MovieRecommender()




if __name__ == "__main__":
    rec = MovieRecommender()
    
    try:
        rec.load_data('TMDB 5000 Movies.csv')

        print("\n" + "="*50)
        print("DIEGE: SITUATIONAL MOVIE DISCOVERY")
        print("Type 'quit' or 'exit' to stop.")
        print("="*50)

        while True:
            user_prompt = input("\n Describe your current situation or feeling for the perfect movies: ")
            if user_prompt.lower() in ['quit', 'exit', 'q']:
                print("Closing Diege. Hope you found your perfect movie!")
                break
            if not user_prompt.strip():
                continue
            print("Searching the diegesis...")
            results = rec.get_vibe_match(user_prompt)

            for i, movie in enumerate(results, 1):
                print(f"{i}. {movie['title']}")
                print(f"   Summary: {movie['vibe_features'][:100]}...") 
                print("-" * 30)

        # test_prompt = "I'm feeling lonely and want to watch something about space exploration"
        # print(f"\n--- Testing Vibe: '{test_prompt}' ---")
        
        # results = rec.get_vibe_match(test_prompt)
        
        # for i, movie in enumerate(results, 1):
        #     print(f"{i}. {movie['title']}")
        #     print(f"   Summary: {movie['overview'][:100]}...") 
        #     print("-" * 30)
            
    except FileNotFoundError:
        print("Error: Could not find 'tmdb_5000_movies.csv'. Make sure it's in the backend folder.")

