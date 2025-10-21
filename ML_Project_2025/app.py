import cv2
import numpy as np
from deepface import DeepFace
from collections import deque, Counter
from recommender import recommend_songs

emotion_window = deque(maxlen=15)
frame_count = 0
analyze_every = 10
stable_emotion = "unknown"
last_recommended_emotion = None
current_songs = []

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % analyze_every != 0:
        # Display stable emotion and current songs
        cv2.putText(frame, f"Emotion: {stable_emotion}", (30,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        for i, song in enumerate(current_songs):
            y = 80 + i*30
            text = f"{i+1}. {song['track_name']} - {song['artist_name']}"
            cv2.putText(frame, text, (30, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
        cv2.imshow("Emotion Music Recommender", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    try:
        h, w = frame.shape[:2]
        min_dim = min(h, w)
        cropped = frame[(h-min_dim)//2:(h+min_dim)//2, (w-min_dim)//2:(w+min_dim)//2]
        small_frame = cv2.resize(cropped, (224, 224))

        kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
        small_frame = cv2.filter2D(small_frame, -1, kernel)

        result = DeepFace.analyze(
            img_path=small_frame,
            actions=['emotion'],
            detector_backend='mtcnn',
            enforce_detection=True,
            silent=True
        )

        if isinstance(result, list):
            result = result[0]

        emotions = result['emotion']
        top_emotion = max(emotions, key=emotions.get)
        top_score = emotions[top_emotion]

        if top_score > 0.4:
            emotion_window.append(top_emotion)

        if emotion_window:
            stable_emotion = Counter(emotion_window).most_common(1)[0][0]

        # Update song list if emotion changed
        if stable_emotion != last_recommended_emotion:
            current_songs = recommend_songs(stable_emotion)
            last_recommended_emotion = stable_emotion

    except Exception as e:
        print(f"⚠️ Error: {e}")

    # Overlay emotion and songs
    cv2.putText(frame, f"Emotion: {stable_emotion}", (30,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    for i, song in enumerate(current_songs):
        y = 80 + i*30
        text = f"{i+1}. {song['track_name']} - {song['artist_name']}"
        cv2.putText(frame, text, (30, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

    cv2.imshow("Emotion Music Recommender", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
