from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
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

# Import voice system

from .test_voice import start_voice_analysis, voice_metrics, reset_voice_metrics

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------

# LOAD EMOTION MODEL

# -----------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = "/Users/mayanksmac/Desktop/DL/models/emotion_model.h5"

model = load_model(model_path, compile=False)

class_names = [
"angry", "disgust", "fear",
"happy", "neutral", "sad", "surprise"
]

# -----------------------------

# FACE DETECTOR

# -----------------------------

face_cascade = cv2.CascadeClassifier(
cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# -----------------------------

# START VOICE THREAD

# -----------------------------

threading.Thread(target=start_voice_analysis, daemon=True).start()

# -----------------------------

# GLOBAL STATE

# -----------------------------

emotion_buffer = deque(maxlen=5) # Reduced from 10 to 5 for faster "immediate" feedback

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

# Helper to load/save history
def get_history():
    history_path = os.path.join(os.path.dirname(BASE_DIR), "results.json")
    if os.path.exists(history_path):
        try:
            with open(history_path, "r") as f:
                return json.load(f)
        except:
            return []
    return []

def save_session_to_history(session_data):
    history = get_history()
    # If history is a dict (old format), convert to list
    if isinstance(history, dict):
        history = [history]
    history.append(session_data)
    history_path = os.path.join(os.path.dirname(BASE_DIR), "results.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)

@app.get("/history")
async def history_endpoint():
    return get_history()

# -----------------------------

# EMOTION PREDICTION

# -----------------------------

def predict_emotion(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(60,60)
    )

    if len(faces) == 0:
        return "No Face", 0, 0

    x, y, w, h = faces[0]

    face = gray[y:y+h, x:x+w]

    face = cv2.resize(face, (48,48))
    face = face.astype("float32") / 255.0
    face = np.expand_dims(face, axis=-1)
    face = np.expand_dims(face, axis=0)

    prediction = model.predict(face, verbose=0)[0]

    emotion_buffer.append(prediction)

    avg_prediction = np.mean(emotion_buffer, axis=0)

    emotion = class_names[np.argmax(avg_prediction)]

    confidence = float(np.max(avg_prediction))

    happy = avg_prediction[class_names.index("happy")]
    surprise = avg_prediction[class_names.index("surprise")]
    neutral = avg_prediction[class_names.index("neutral")]

    engagement = int((happy*1.0 + surprise*0.9 + neutral*0.6) * 100)

    stability_from_face = int(confidence * 100)

    return emotion, engagement, stability_from_face

# -----------------------------

# WEBSOCKET

# -----------------------------

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # Initialize fresh session state for this connection
    local_selected_questions = [first_question] + random.sample(other_questions, 10)
    local_current_question_index = 0
    total_session_start_time = time.time()
    local_question_start_time = time.time()
    is_completed = False
    
    # Track metrics for each question
    session_history = []
    
    # Reset voice metrics for a fresh start
    reset_voice_metrics()
    
    # Running averages/counters for the current question
    current_q_metrics = {
        "wpm_list": [],
        "engagement_list": [],
        "filler_count_start": voice_metrics["filler_count"]
    }

    print("Client connected")

    # Send first question immediately
    await websocket.send_json({
        "question": local_selected_questions[local_current_question_index],
        "time": 0,
        "wpm": 0,
        "fillers": 0,
        "stability": 0,
        "emotion": "Neutral",
        "engagement": 0,
        "completed": False,
        "is_last_question": local_current_question_index == len(local_selected_questions) - 1
    })

    try:

        while True:
            message = await websocket.receive_text()
            data = json.loads(message)

            # NEXT QUESTION
            if data.get("type") == "next":
                # Save current question metrics before moving on
                final_q_wpm = np.mean(current_q_metrics["wpm_list"]) if current_q_metrics["wpm_list"] else 0
                final_q_engagement = np.mean(current_q_metrics["engagement_list"]) if current_q_metrics["engagement_list"] else 0
                final_q_fillers = voice_metrics["filler_count"] - current_q_metrics["filler_count_start"]
                final_q_stability = int(voice_metrics["volume_stability"] * 100)
                
                session_history.append({
                    "question": local_selected_questions[local_current_question_index],
                    "duration": int(time.time() - local_question_start_time),
                    "avg_wpm": int(final_q_wpm),
                    "avg_engagement": int(final_q_engagement),
                    "fillers": final_q_fillers,
                    "stability": final_q_stability
                })

                if local_current_question_index < len(local_selected_questions)-1:
                    local_current_question_index += 1
                    local_question_start_time = time.time()
                    # Reset current question metrics
                    current_q_metrics = {
                        "wpm_list": [],
                        "engagement_list": [],
                        "filler_count_start": voice_metrics["filler_count"]
                    }
                else:
                    is_completed = True
                    # Final results saving logic
                    final_results = {
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "total_duration": int(time.time() - total_session_start_time),
                        "overall_avg_wpm": int(np.mean([h["avg_wpm"] for h in session_history])),
                        "overall_avg_engagement": int(np.mean([h["avg_engagement"] for h in session_history])),
                        "overall_avg_stability": int(np.mean([h["stability"] for h in session_history])),
                        "total_fillers": sum([h["fillers"] for h in session_history]),
                        "filler_freq": voice_metrics.get("filler_freq", {}),
                        "history": session_history
                    }
                    
                    save_session_to_history(final_results)

                    await websocket.send_json({
                        "question": "Interview Completed",
                        "completed": True,
                        "report": final_results
                    })

                    continue

            # FRAME RECEIVED
            if data.get("type") == "frame":

                try:

                    if "image" not in data:
                        continue

                    image_parts = data["image"].split(",")

                    if len(image_parts) < 2:
                        continue

                    image_data = image_parts[1]

                    decoded = base64.b64decode(image_data)

                    if len(decoded) == 0:
                        continue

                    np_arr = np.frombuffer(decoded, np.uint8)

                    if np_arr.size == 0:
                        continue

                    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                    if frame is None:
                        continue

                except Exception as e:
                    print("Frame decode error:", e)
                    continue

                emotion, engagement, face_stability = predict_emotion(frame)
                
                # Update current question trackers
                current_q_metrics["wpm_list"].append(voice_metrics["wpm"])
                current_q_metrics["engagement_list"].append(engagement)

                # Use per-question timer during interview, total timer at the end
                if is_completed:
                    elapsed = int(time.time() - total_session_start_time)
                else:
                    elapsed = int(time.time() - local_question_start_time)

                safe_stability = voice_metrics["volume_stability"]
                safe_stability = 0 if (safe_stability != safe_stability) else safe_stability  # NaN check

                safe_wpm = voice_metrics["wpm"]
                safe_wpm = 0 if (safe_wpm != safe_wpm) else safe_wpm  # NaN check

                await websocket.send_json({

                    "question": "Interview Completed" if is_completed else local_selected_questions[local_current_question_index],

                    "time": elapsed,

                    "wpm": round(safe_wpm, 1),

                    "fillers": voice_metrics["filler_count"],

                    "stability": int(safe_stability * 100),

                    "emotion": emotion,

                    "engagement": engagement,

                    "completed": is_completed,
                    
                    "is_last_question": local_current_question_index == len(local_selected_questions) - 1
                })

    except WebSocketDisconnect:

        print("Client disconnected")