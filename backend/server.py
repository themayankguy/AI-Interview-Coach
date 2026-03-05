from fastapi import FastAPI, WebSocket
import time
import json
import base64
import numpy as np
import cv2
from collections import deque
from tensorflow.keras.models import load_model
import os
import random
import threading

# 🔥 Import your real voice system
from test_voice import start_voice_analysis, voice_metrics

app = FastAPI()

# -----------------------------
# LOAD EMOTION MODEL
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = "/Users/mayanksmac/Desktop/DL/models/emotion_model.h5"

model = load_model(model_path, compile=False)

class_names = [
    'angry', 'disgust', 'fear',
    'happy', 'neutral', 'sad', 'surprise'
]

# -----------------------------
# FACE DETECTOR
# -----------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# -----------------------------
# START VOICE THREAD (ONLY ONCE)
# -----------------------------
threading.Thread(target=start_voice_analysis, daemon=True).start()

# -----------------------------
# GLOBAL STATE
# -----------------------------
emotion_buffer = deque(maxlen=10)

first_question = "Tell me about yourself."

other_questions = [
    # HR / Behavioral
    "Why should we hire you?",
    "What are your strengths and weaknesses?",
    "Describe a challenge you handled.",
    "Where do you see yourself in 5 years?",
    "Tell me about a failure you experienced.",
    "How do you handle pressure?",
    "Tell me about a time you worked in a team.",
    "How do you resolve conflicts in a project?",
    "What motivates you to work in AI & Data Science?",
    "Why do you want to join our company?",

    # Core AI & Data Science (BTech AIDS)
    "What is the difference between AI, ML, and Deep Learning?",
    "Explain supervised vs unsupervised learning.",
    "What is overfitting and how can you prevent it?",
    "Explain bias-variance tradeoff.",
    "What are precision, recall, and F1-score?",
    "What is cross-validation?",
    "Difference between classification and regression?",
    "Explain Gradient Descent and its types.",
    "What are activation functions? Name a few.",
    "What is backpropagation?",

    # Deep Learning
    "What is CNN and where is it used?",
    "What is the role of convolutional layers?",
    "Explain RNN and LSTM.",
    "What is transfer learning?",
    "What are vanishing and exploding gradients?",
    "Difference between Batch Normalization and Dropout?",
    "What is an epoch, batch size, and iteration?",

    # Data Structures & Programming
    "Explain time complexity and space complexity.",
    "What is the difference between list and tuple in Python?",
    "What are decorators in Python?",
    "Explain OOP concepts in Python.",
    "What is multithreading vs multiprocessing?",

    # DBMS & Data Handling
    "What is normalization?",
    "Explain ACID properties.",
    "Difference between SQL and NoSQL?",
    "What is indexing in databases?",
    "What are joins in SQL? Explain types.",

    # Statistics & Mathematics
    "What is probability distribution?",
    "Difference between mean, median, and mode?",
    "Explain standard deviation.",
    "What is correlation vs covariance?",
    "What is Bayes Theorem?",

    # Projects & Practical
    "Explain your final year project.",
    "What challenges did you face in your AI project?",
    "How did you optimize your model performance?",
    "How do you handle imbalanced datasets?",
    "Which tools and frameworks have you used in AI projects?"
]
random.shuffle(other_questions)
selected_questions = [first_question] + random.sample(other_questions, 10)

current_question_index = 0
question_start_time = time.time()

# -----------------------------
# EMOTION PREDICTION
# -----------------------------
def predict_emotion(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(60, 60)
    )

    if len(faces) == 0:
        return "No Face", 0, 0

    x, y, w, h = faces[0]
    face = gray[y:y+h, x:x+w]

    face = cv2.resize(face, (48, 48))
    face = face.astype("float32") / 255.0
    face = np.expand_dims(face, axis=-1)
    face = np.expand_dims(face, axis=0)

    prediction = model.predict(face, verbose=0)[0]

    emotion_buffer.append(prediction)
    avg_prediction = np.mean(emotion_buffer, axis=0)

    emotion = class_names[np.argmax(avg_prediction)]
    confidence = float(np.max(avg_prediction))

    # Real engagement from emotion probabilities
    happy = avg_prediction[class_names.index("happy")]
    surprise = avg_prediction[class_names.index("surprise")]
    neutral = avg_prediction[class_names.index("neutral")]

    engagement = int(
        (happy * 1.0 + surprise * 0.9 + neutral * 0.6) * 100
    )

    stability_from_face = int(confidence * 100)

    return emotion, engagement, stability_from_face


# -----------------------------
# WEBSOCKET
# -----------------------------
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global current_question_index, question_start_time

    await websocket.accept()

    while True:
        message = await websocket.receive_text()
        data = json.loads(message)

        # NEXT QUESTION
        if data.get("type") == "next":
            if current_question_index < len(selected_questions) - 1:
                current_question_index += 1
                question_start_time = time.time()
            else:
                await websocket.send_json({
                    "question": "Interview Completed 🎉",
                    "completed": True
                })
                continue

        # FRAME RECEIVED
        if data.get("type") == "frame":

            image_data = data["image"].split(",")[1]
            decoded = base64.b64decode(image_data)
            np_arr = np.frombuffer(decoded, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            emotion, engagement, face_stability = predict_emotion(frame)

            elapsed = int(time.time() - question_start_time)

            await websocket.send_json({
                "question": selected_questions[current_question_index],
                "time": elapsed,

                # 🔥 REAL VOICE METRICS
                "wpm": voice_metrics["wpm"],
                "fillers": voice_metrics["filler_count"],
                "stability": int(voice_metrics["volume_stability"] * 100),

                # 🔥 REAL EMOTION METRICS
                "emotion": emotion,
                "engagement": engagement,

                "completed": False
            })

# TERMINAL 1
#cd ai-interview/backend
#uvicorn server:app --reload

#Terminal 2
#cd /Users/mayanksmac/Desktop/DL/ai-interview/frontend
#python3 -m http.server 5500

# Open http://localhost:5500 in browser to test