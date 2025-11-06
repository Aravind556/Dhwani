import streamlit as st
import joblib
import pandas as pd
import torch
from PIL import Image
import torchvision.transforms as T
import torchvision.models as models
import torch.nn as nn
import numpy as np
import cv2
import time
from collections import Counter
import os

# -------------------------------
# Load mood prediction models
# -------------------------------
gmm = joblib.load("results/gmm_mood_model.pkl")
scaler = joblib.load("results/scaler.pkl")
pca = joblib.load("results/pca.pkl")
mood_labels = {
    0: "Sad / Mellow",
    1: "Energetic / Party",
    2: "Happy / Chill",
    3: "Calm / Content"
}

# -------------------------------
# Emotion prediction model setup
# -------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 4
TARGET_NAMES = {0: "Sad", 1: "Happy", 2: "Energetic", 3: "Calm"}

def get_model(num_classes=NUM_CLASSES, pretrained=False):
    m = models.efficientnet_b0(weights=None)  # use new weights argument
    for name, param in m.features.named_parameters():
        if "6" in name or "7" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    in_f = m.classifier[1].in_features
    m.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_f, num_classes)
    )
    return m.to(DEVICE)

def get_transforms(train=False):
    return T.Compose([
        T.Resize((224, 224)),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225])
    ])

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
                                     "haarcascade_frontalface_default.xml")

def rigid_preprocess(img_pil):
    """Crop face using Haar Cascade and normalize lighting."""
    img = np.array(img_pil.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda f: f[2]*f[3])
        img = img[y:y+h, x:x+w]
    else:
        # fallback: center crop
        h, w, _ = img.shape
        s = int(min(h, w) * 0.8)
        y1, x1 = (h - s) // 2, (w - s) // 2
        img = img[y1:y1 + s, x1:x1 + s]
    # lighting normalization
    img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    return Image.fromarray(img)

def predict_from_image(model, img, transform=None, threshold=0.4):
    """Predict emotion label with softmax confidence."""
    if isinstance(img, np.ndarray):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    t = transform or get_transforms(False)
    xb = t(img).unsqueeze(0).to(DEVICE)
    model.eval()
    with torch.no_grad():
        out = model(xb)
        probs = torch.nn.functional.softmax(out, dim=1)
        conf, idx = torch.max(probs, dim=1)
        conf_val = conf.item()
        idx_val = idx.item()
    if conf_val >= threshold:
        label = TARGET_NAMES[idx_val]
        return f"{label} ({conf_val*100:.1f}%)"
    return "Calm"

# -------------------------------
# Load emotion model safely
# -------------------------------
@st.cache_resource
def load_emotion_model():
    model = get_model(pretrained=False)
    model_path = os.path.join(os.path.dirname(__file__),
                              "results", "fine_tunedme2_sad_effnetb0.pth")
    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found: {model_path}")
        return None
    try:
        state_dict = torch.load(model_path, map_location=DEVICE)
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        model.eval()
        st.success("✅ Emotion detection model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        return None

emotion_model = load_emotion_model()

# -------------------------------
# Load songs data
# -------------------------------
songs_df = pd.read_csv("Dataset/songs_with_moods_fixed.csv")
emotion_to_mood = {
    "Sad": "Sad / Mellow",
    "Happy": "Happy / Chill",
    "Energetic": "Energetic / Party",
    "Calm": "Calm / Content"
}

def get_songs_by_mood(mood):
    mood = mood.lower()
    results = songs_df[songs_df['mood_label'].str.lower().str.contains(mood)]
    return results

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🎵 Mood & Emotion-Based Song Recommender")
st.caption("Real-time emotion recognition + song mood clustering integration")

# --- Section 1: Mood prediction ---
st.header("1️⃣ Predict Mood from Song Features")
valence = st.slider("Valence", 0.0, 1.0, 0.5)
energy = st.slider("Energy", 0.0, 1.0, 0.5)
danceability = st.slider("Danceability", 0.0, 1.0, 0.5)
tempo = st.slider("Tempo", 0.0, 250.0, 120.0)

mood_pred = None
if st.button("🎯 Predict Mood"):
    df = pd.DataFrame([[valence, energy, danceability, tempo]],
                      columns=["valence", "energy", "danceability", "tempo"])
    X_scaled = scaler.transform(df)
    X_reduced = pca.transform(X_scaled)
    cluster = gmm.predict(X_reduced)[0]
    mood_pred = mood_labels[cluster]
    st.success(f"Predicted Mood: **{mood_pred}**")

# --- Section 2: Webcam emotion detection ---
st.header("2️⃣ Real-Time Emotion Detection")
st.write("Enable webcam to detect your facial emotion and get personalized song suggestions.")
run = st.checkbox("🎥 Start Webcam")
frame_placeholder = st.empty()
emotion_placeholder = st.empty()
recommendations_placeholder = st.empty()
emotion_history = []
last_recommendation_time = time.time()

if run and emotion_model:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Could not access webcam.")
    else:
        try:
            while run:
                ret, frame = cap.read()
                if not ret or frame is None:
                    st.warning("⚠️ Skipping invalid frame.")
                    continue
                if not hasattr(frame, "shape") or len(frame.shape) != 3 or frame.shape[0] == 0 or frame.shape[1] == 0:
                    st.warning("⚠️ Empty frame detected, skipping.")
                    continue

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_frame)
                processed_img = rigid_preprocess(pil_img)
                emotion_full = predict_from_image(emotion_model, processed_img)
                emotion = emotion_full.split(" ")[0]
                emotion_history.append(emotion)

                cv2.putText(frame, emotion_full, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                frame_placeholder.image(frame, channels="BGR", use_container_width=True)
                emotion_placeholder.text(f"Emotion: {emotion_full}")

                # Every 30 seconds → recommend songs
                current_time = time.time()
                if current_time - last_recommendation_time >= 30:
                    if emotion_history:
                        most_common_emotion = Counter(emotion_history).most_common(1)[0][0]
                        emotion_history.clear()

                        mapped_mood = emotion_to_mood.get(
                            most_common_emotion, "Calm / Content")
                        recommendations = []

                        # Include mood-based + emotion-based mix
                        if mood_pred:
                            mood_results = get_songs_by_mood(mood_pred)
                            mood_songs = mood_results['song_name'].sample(
                                n=min(2, len(mood_results))).tolist()
                            recommendations.extend(mood_songs)

                        emotion_results = get_songs_by_mood(mapped_mood)
                        emotion_songs = emotion_results['song_name'].sample(
                            n=min(3, len(emotion_results))).tolist()
                        recommendations.extend(emotion_songs)

                        recommendations_placeholder.markdown(
                            f"**🎵 Recommendations (Mood: {mood_pred or 'None'}, Emotion: {most_common_emotion})**\n" +
                            "\n".join([f"- {song}" for song in recommendations])
                        )
                    last_recommendation_time = current_time

                time.sleep(0.1)
        finally:
            cap.release()
            cv2.destroyAllWindows()

# --- Section 3: Summary ---
st.header("3️⃣ Combined Status")
if mood_pred:
    st.info(f"🎧 Current Mood: {mood_pred}")
else:
    st.write("ℹ️ Predict mood above to enhance recommendations.")
st.write("Real-time emotion updates every frame. Song suggestions refresh every 30s.")

