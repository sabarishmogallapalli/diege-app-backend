import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import ast

def clean_json_list(json_str):
    if not json_str or json_str == '[]': return ""
    try:
        data = ast.literal_eval(json_str)
        return " ".join([i['name'] for i in data])
    except:
        return ""

print("Loading dataset...")
df = pd.read_csv('TMDB 5000 Movies.csv')
df = df[['id', 'title', 'overview', 'genres', 'keywords', 'tagline']].fillna('')

df['genres'] = df['genres'].apply(clean_json_list)
df['keywords'] = df['keywords'].apply(clean_json_list)
df['vibe_features'] = (df['overview'] + " " + df['genres'] + " " + df['keywords'] + " " *2 + df['tagline'] *2).str.lower()

print("Baking the Diege Brain (This will take a minute on your Mac)...")
encoder = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = encoder.encode(df['vibe_features'].tolist(), show_progress_bar=True)

# Save the mathematical vectors to a highly compressed file
np.save('movie_embeddings.npy', embeddings)
print("Brain successfully baked and saved as 'movie_embeddings.npy'!")