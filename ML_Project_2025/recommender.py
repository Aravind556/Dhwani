import pandas as pd

# Load dataset
df = pd.read_csv("data/SpotifyFeatures.csv")

def recommend_songs(emotion, n=3):
    emotion = emotion.lower()

    # Map emotions to audio features
    if emotion == 'happy':
        subset = df[(df['valence'] > 0.6) & (df['energy'] > 0.6)]
    elif emotion == 'sad':
        subset = df[(df['valence'] < 0.4) & (df['energy'] < 0.5)]
    elif emotion == 'angry':
        subset = df[(df['energy'] > 0.7) & (df['valence'] < 0.5)]
    else:  # neutral/calm/unknown
        subset = df[(df['valence'] >= 0.4) & (df['valence'] <= 0.6)]

    # If no match, pick random songs
    if subset.empty:
        subset = df.sample(n)

    picks = subset.sample(min(n, len(subset)))

    return picks[['track_name', 'artist_name']].to_dict(orient='records')
