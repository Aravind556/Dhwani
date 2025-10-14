from fastapi import FastAPI
import joblib
import pandas as pd
from pydantic import BaseModel

app = FastAPI()

# Load models once at startup
gmm = joblib.load("gmm_mood_model.pkl")
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")
mood_labels = {0: "Sad / Mellow", 1: "Energetic / Party", 2: "Happy / Chill", 3: "Calm / Content"}

class SongFeatures(BaseModel):
    valence: float
    energy: float
    danceability: float
    tempo: float

@app.post("/predict-mood")
def predict_mood(song: SongFeatures):
    df = pd.DataFrame([song.dict()])
    X_scaled = scaler.transform(df)
    X_reduced = pca.transform(X_scaled)
    cluster = gmm.predict(X_reduced)[0]
    mood = mood_labels[cluster]
    return {"mood_label": mood}
