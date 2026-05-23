from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from recommender import recommender as rec

app = FastAPI(title="Diege API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows any website to talk to your API
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def load_data():
    print("Diege is waking up...")
    rec.load_data('TMDB 5000 Movies.csv')

class VibeRequest(BaseModel):
    prompt: str

@app.get("/")
def health_check():
    return {"status": "online", "model": "all-MiniLM-L6-v2"}

@app.post("/recommend")
async def get_recommendation(request: VibeRequest):
    try:
        results = rec.get_vibe_match(request.prompt)
        return {"status": "success", "results": results}
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)