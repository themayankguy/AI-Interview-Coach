# AI Interview Coach

A real-time AI-powered mock interview platform that analyzes your **speech, facial emotions, visual engagement, and talking pace** as you answer questions — then generates a comprehensive performance report with charts, AI feedback, and a placement likelihood score.

---

## The Real-World Problem It Solves

Every year, millions of job seekers — especially fresh graduates and early-career professionals — struggle with interviews not because they lack knowledge, but because they lack **interview self-awareness**. They don't realise they speak too fast, use too many filler words, avoid eye contact, or convey nervousness through their expressions. Traditional preparation methods (reading guides, mock interviews with friends, or reviewing past rejection feedback) are **subjective, infrequent, and non-measurable**.

At the same time, hiring companies report that **communication skills, confidence, and body language** are among the top reasons candidates fail interviews — even technically strong ones.

### What This Project Addresses

| Problem | How This Project Solves It |
|---|---|
| No objective feedback on speaking pace | Measures real-time WPM using live speech transcription (Whisper) |
| Unaware of filler word overuse | Counts and tracks "um", "uh", "like", "you know" per session |
| Can't see how nervous they appear | Classifies facial emotion every frame using a trained CNN model |
| No structured practice environment | Provides a curated set of real HR, behavioural, and technical questions |
| No measurable progress over time | Stores all past sessions with scores and charts in session history |
| Expensive coaching / no access to mentors | Runs entirely locally — zero cost, zero subscription, no internet required for inference |
| Interview feedback comes too late (post-rejection) | Delivers real-time metrics *during* the session and a full report immediately after |

### Who Benefits

- **Students** preparing for campus placements and internship interviews
- **Early-career professionals** switching domains or companies
- **Self-learners** who want structured, data-driven interview practice without a coach
- **Educators and institutions** who want a tool for interview readiness training

---

## Features

- 🎙️ **Live Voice Analysis** — Talking speed (WPM), voice stability, and filler word detection via Whisper + sounddevice
- 😐 **Facial Emotion Detection** — Real-time emotion classification (angry, happy, neutral, sad, surprise, etc.) using a custom TensorFlow CNN model
- 👁️ **Visual Engagement Tracking** — Face presence detection via OpenCV Haar cascade throughout the session
- ❓ **Dynamic Interview Questions** — HR, behavioral, and technical questions served in sequence via WebSocket
- 📊 **Performance Report** — Radar chart, score contribution donut, WPM timeline, engagement trend, question-by-question breakdown table
- 🎓 **Grading System** — O / A+ / A / B / C / F scale with percentage scores
- 🏢 **Placement Likelihood Predictor** — AI-assessed probability with a written justification
- 💬 **AI Feedback** — Bullet-point performance feedback generated per session
- 🖨️ **Print Report** — Clean A4-formatted printout (all 4 charts, no dark backgrounds)
- 🗂️ **Session History** — Past sessions stored in `results.json` and viewable in-app
- 🎨 **Silver Surfer UI** — Glassmorphism dark design with Inter font, slate-gradient background, blue→purple gradient accents

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Vanilla HTML/CSS/JS, Chart.js, Inter (Google Fonts) |
| Backend | FastAPI (Python), WebSockets |
| Voice | Whisper (faster-whisper), sounddevice, librosa |
| Vision | OpenCV (cv2), TensorFlow/Keras (custom emotion model) |
| Serving | `npx serve` (frontend), `uvicorn` (backend) |

---

## Project Structure

```
DL/
├── backend/
│   ├── server.py          # FastAPI WebSocket server — main backend
│   ├── test_voice.py      # Voice analysis thread (Whisper + librosa)
│   ├── test_facial.py     # Facial emotion testing utility
│   └── train_facial.py    # Emotion model training script
├── frontend/
│   └── index.html         # Single-page app (all UI + JS)
├── models/
│   └── emotion_model.h5   # Trained Keras emotion classification model
├── results.json           # Session history (auto-updated after each session)
├── main.py                # Entry point / misc runner
└── README.md
```

---

## Prerequisites

- Python 3.10+
- Node.js (for `npx serve`)
- A working **microphone** and **webcam**

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/themayankguy/AI-Interview-Coach.git
cd AI-Interview-Coach
```

### 2. Create a Python virtual environment

```bash
python3.10 -m venv venv310
source venv310/bin/activate
```

### 3. Install Python dependencies

```bash
pip install fastapi uvicorn websockets opencv-python tensorflow \
            librosa sounddevice faster-whisper numpy
```

> **Note:** `faster-whisper` requires `ffmpeg`. Install via Homebrew on macOS:
> ```bash
> brew install ffmpeg
> ```

### 4. Ensure the emotion model is in place

The trained model should be at:
```
models/emotion_model.h5
```

If missing, retrain using:
```bash
python backend/train_facial.py
```

---

## Running the Project

You need **two terminals** running simultaneously.

### Terminal 1 — Start the Backend

From the project root:

```bash
source venv310/bin/activate
python3 -m uvicorn backend.server:app --port 8000 --reload
```

The backend WebSocket server will be available at:
```
ws://localhost:8000/ws
```

### Terminal 2 — Start the Frontend

From the project root:

```bash
source venv310/bin/activate
python3 -m http.server 8001
```

Then open your browser at:
```
http://localhost:8001/frontend/index.html
```

---

## Usage

1. Open the frontend URL in your browser
2. Allow **camera** and **microphone** access when prompted
3. Click **Begin Interview** to start the session
4. Answer each question — the AI analyses your speech and face in real-time
5. Click **Next Question** to move to the next one
6. After the final question, the full **Performance Report** appears automatically
7. Use **Print Report** to export a clean A4 PDF

---

## Notes

- The backend must be running **before** you click Begin Interview — otherwise the WebSocket connection will fail
- Session results are saved to `results.json` after each completed session
- The emotion model path is currently hardcoded in `server.py` — update `model_path` if you move the model
- Print works best in **Chrome** or **Edge** with "Background graphics" disabled in print settings

---

## Deep Learning & Computer Vision — How the CNN Works

One of the core AI components of this project is a **Convolutional Neural Network (CNN)** trained to classify human facial emotions in real-time. This is where **Deep Learning** and **Computer Vision** intersect.

### What is a CNN?

A **Convolutional Neural Network** is a class of deep neural network specifically designed to process grid-structured data like images. Unlike a regular fully-connected network, a CNN learns **spatial hierarchies of features** — from simple edges and textures in early layers, to complex structures like eyes, noses, and expressions in deeper layers.

A CNN typically consists of:

| Layer Type | Role |
|---|---|
| **Convolutional Layer** | Applies learned filters (kernels) across the image to detect local features (edges, textures) |
| **ReLU Activation** | Introduces non-linearity so the network can learn complex patterns |
| **Pooling Layer** | Downsamples the feature maps, reducing spatial size and computation |
| **Flatten + Dense Layers** | Converts 2D feature maps into a 1D vector and produces the final class probabilities |
| **Softmax Output** | Outputs a probability for each emotion class |

### How It's Used Here

When a video frame arrives at the backend (`server.py`):

1. **OpenCV** (`cv2`) detects the face region using a Haar Cascade classifier — this is the Computer Vision step
2. The detected face crop is **resized to 48×48 pixels** and **normalised** (pixel values 0–1)
3. The 48×48 grayscale image is fed into the trained **Keras CNN** (`emotion_model.h5`)
4. The model outputs a probability distribution across 7 emotion classes: `angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`
5. The highest-probability class is selected as the **detected emotion** and sent to the frontend via WebSocket in real-time

### Connection to Deep Learning

This project is a practical application of **Deep Learning** because:

- The CNN was **trained end-to-end** on a labelled facial expression dataset (e.g., FER-2013), learning directly from raw pixel data without hand-crafted feature engineering
- The model uses **multiple stacked convolutional layers**, each learning progressively more abstract representations — this is the hallmark of *deep* learning
- **Backpropagation** and the **Adam optimizer** were used to minimise cross-entropy loss across the 7-class classification problem

### Connection to Computer Vision

Computer Vision is the field of enabling machines to interpret and understand visual information. This project applies CV at two stages:

1. **Face Detection** — Using OpenCV's Haar Cascade to locate faces in each frame (classical CV)
2. **Emotion Recognition** — Using the CNN to classify the detected face into an emotion category (deep learning-based CV)

Together, they form a real-time **visual intelligence pipeline** that runs at every frame of the interview, feeding emotion and engagement data into the final performance score.

---

## How AI Feedback & Placement Likelihood Work

### AI Feedback

The **AI Feedback** section is generated entirely in the **frontend** using rule-based threshold logic (`generateAIFeedback()` in `index.html`). No external AI model or API is called. At the end of the session, four final metrics are evaluated against fixed thresholds and a corresponding feedback sentence is selected:

| Metric | Threshold | What Gets Said |
|---|---|---|
| **Visual Presence** (engagement %) | `< 60` → low, `60–80` → moderate, `≥ 80` → strong | Feedback on eye contact and facial expressiveness |
| **Talking Speed** (WPM) | `< 100` → too slow, `> 190` → too fast, else → ideal | Feedback on speech pace and delivery |
| **Filler Words** (count) | `> 10` → heavy, `1–10` → minor, `0` → perfect | Feedback on speech clarity and hesitation |
| **Voice Stability** (%) | `< 60` → unstable, `≥ 60` → steady | Feedback on vocal confidence and authority |

These four pre-written sentences are then rendered as bullet points in the report. The word "AI" here refers to the system as a whole — the feedback logic itself is handcrafted conditional logic, not a generative model.

> **Future scope:** The feedback could be made genuinely AI-generated by sending the session metrics to an LLM API (e.g., Gemini, GPT-4) and requesting a contextual, personalised response.

---

### Placement Likelihood

The **Placement Likelihood %** is computed in `calculatePlacementLikelihood()` in the frontend. It is **not predicted by any ML model**. The process is:

1. The session's weighted score is converted to a **grade** (O / A+ / A / B / C / F)
2. Based on the grade, a random number is generated within a fixed band:

| Grade | Probability Range |
|---|---|
| O | 90 – 98% |
| A+ | 80 – 89% |
| A | 70 – 79% |
| B | 50 – 69% |
| C | 30 – 49% |
| F | 10 – 29% |

3. A **hardcoded justification string** for that grade band is displayed alongside the percentage

Because `Math.random()` is used, the displayed percentage will differ on every page load even for the same session data. The justification text is fixed per grade.

> **Future scope:** A regression model trained on real placement outcome data (scores ↔ offer received/not) could replace this with a genuine, data-driven probability prediction.