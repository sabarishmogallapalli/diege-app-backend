import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import numpy as np
import ast
from transformers import pipeline

print("Hello World")


def clean_json_list(json_str):
        if not json_str or json_str == '[]':
            return ""
        try:
            # Converts string '[{"name": "Action"}]' to list and extracts names
            data = ast.literal_eval(json_str)
            return " ".join([i['name'] for i in data])
        except:
            return ""

class MovieRecommender:

    def __init__(self):
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        #parameters I declare
        self.classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        self.tmdb = None
        self.embeddings = None
        self.knn = None
        
    def get_vibe_match(self, user_prompt):
        if len(user_prompt) < 4:
            return []
        user_vector = self.encoder.encode([user_prompt])
        distances, indices = self.knn.kneighbors(user_vector)
        mood_map = {
            "Epic": ["Action"],
            "Spy": ["Action"],
            "Disaster": ["Action"],
            "Superhero": ["Action"],
            "Thriller": ["Action"],
            "Martial Arts": ["Action"],
            "Video Game": ["Action"],
            "Whodunnit": ["Crime"],
            "Detective": ["Crime"],
            "Gangster": ["Crime"],
            "Hardboiled": ["Crime"],
            "Courtroom": ["Crime"],
            "Legal": ["Crime"],
            "Contemporary Fantasy": ["Fantasy"],
            "Urban Fantasy": ["Fantasy"],
            "Dark Fantasy": ["Fantasy"],
            "Fairy Tale": ["Fantasy"],
            "Epic Fantasy": ["Fantasy"],
            "Heroic Fantasy": ["Fantasy"],
            "Sword and Sorcery": ["Fantasy"],
            "Spaghetti Western": ["Western"],
            "Epic Western": ["Western"],
            "Outlaw Western": ["Western"],
            "Marshal Western": ["Western"],
            "Revisionist Western": ["Western"],
            "Revenge Western": ["Western"],
            "Empire Western": ["Western"],
            "Biopic": ["History"],
            "Historical Drama": ["History"],
            "Biblical": ["History"],
            "Period": ["History"],
            "Alternate History": ["History"],
            "Romantic Drama": ["Romance"],
            "Rom-Com": ["Romance"],
            "Chick Flick": ["Romance"],
            "Romantic Thriller": ["Romance"],
            "Traditional Animation": ["Animation"],
            "Rotoscoping": ["Animation"],
            "Puppet Animation": ["Animation"],
            "Claymation": ["Animation"],
            "Live Action/Animation": ["Animation"],
            "Cutout Animation": ["Animation"],
            "2D CGI Animation": ["Animation"],
            "3D CGI Animation": ["Animation"],
            "Slasher": ["Horror"],
            "Splatter": ["Horror"],
            "Psychological Horror": ["Horror"],
            "Survival Horror": ["Horror"],
            "Found Footage": ["Horror"],
            "Paranormal/Occult Horror": ["Horror"],
            "Monster": ["Horror"],
            "Hard Sci-Fi": ["Science Fiction"],
            "Apocalyptic Sci-Fi": ["Science Fiction"],
            "Future Noir": ["Science Fiction"],
            "Space Opera": ["Science Fiction"],
            "Military Science Fiction": ["Science Fiction"],
            "Punk Sci-Fi": ["Science Fiction"],
            "Speculative Sci-Fi": ["Science Fiction"],
        }
        mood_result = self.classifier(user_prompt, candidate_labels=list(mood_map.keys()))
        top_mood = mood_result['labels'][0]
        target_genres = mood_map[top_mood]
        results = []
        threshold = 0.8
        for dist, i in zip(distances[0], indices[0]):
            movie = self.df.iloc[i]
            movie_genres = movie['genres']

            if ("reflective" in top_mood or "Fairy Tale" in top_mood) and "Horror" in movie_genres:
                dist += 0.6
            
            if any(tg in movie_genres for tg in target_genres):
                dist -= 0.1

            if dist < threshold:
                results.append({
                    "id": int(movie['id']),
                    "title": movie['title'],
                    "overview": movie['overview'],
                    "vibe_match": top_mood
                    # "vibe_features": movie['vibe_features']
                })
        return results
    
    def load_data(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.df = self.df[['id', 'title', 'overview', 'genres', 'keywords', 'tagline']].fillna('')

        self.df['genres'] = self.df['genres'].apply(clean_json_list)
        self.df['keywords'] = self.df['keywords'].apply(clean_json_list)

        #THE ULTIMATE DATA ANALYSIS PROMPT
        self.df['vibe_features'] = (
            self.df['overview'] + " " + 
            self.df['genres'] + " " + 
            self.df['keywords'] + " " *2 + 
            self.df['tagline'] *2
        ).str.lower()

        print("encoding the Movie Vibes")
        self.embeddings = self.encoder.encode(self.df['vibe_features'].tolist(), show_progress_bar=True)
        self.knn = NearestNeighbors(n_neighbors=6, metric='cosine')
        self.knn.fit(self.embeddings)
        self.df['genres'] = self.df['genres'].apply(clean_json_list)
        self.df['keywords'] = self.df['keywords'].apply(clean_json_list)
        self.df['vibe_features'] = self.df['overview'] + " " + self.df['genres'] + " " + self.df['keywords']

        print("Diege Brain is ready.")
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

