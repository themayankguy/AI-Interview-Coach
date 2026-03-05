import cv2
import numpy as np
from collections import deque
from tensorflow.keras.models import load_model

def main():

    model = load_model("models/emotion_model.h5")
    print("Model loaded successfully.")

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    cap = cv2.VideoCapture(0)

    class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

    emotion_buffer = deque(maxlen=15)

    total_engagement = 0
    frame_count = 0

    # Default text (so it never crashes)
    avg_session_engagement = 0

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=8,
            minSize=(60, 60)
        )

        for (x, y, w, h) in faces:

            face = gray[y:y+h, x:x+w]
            face = cv2.equalizeHist(face)
            face = cv2.resize(face, (48, 48))
            face = face.astype("float32") / 255.0
            face = np.expand_dims(face, axis=-1)
            face = np.expand_dims(face, axis=0)

            prediction = model.predict(face, verbose=0)[0]

            emotion_buffer.append(prediction)
            avg_prediction = np.mean(emotion_buffer, axis=0)

            confidence = float(np.max(avg_prediction))
            emotion_index = np.argmax(avg_prediction)
            emotion_label = class_names[emotion_index]

            engagement_score = (
                avg_prediction[class_names.index('happy')] * 1.0 +
                avg_prediction[class_names.index('surprise')] * 0.9 +
                avg_prediction[class_names.index('neutral')] * 0.6
            )

            engagement_score = min(engagement_score, 1.0)

            total_engagement += engagement_score
            frame_count += 1
            avg_session_engagement = total_engagement / frame_count

            label_text = f"{emotion_label} ({confidence*100:.1f}%)"
            engagement_text = f"Engagement: {engagement_score*100:.0f}%"

            # Face box
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Background box
            cv2.rectangle(frame, (x, y-70), (x+w, y-10), (0, 0, 0), -1)

            cv2.putText(frame, label_text,
                        (x+5, y-45),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)

            cv2.putText(frame, engagement_text,
                        (x+5, y-20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 0), 2)

        # ---------- ALWAYS DRAW AVERAGE (outside loop) ----------
        avg_text = f"Average Engagement: {avg_session_engagement*100:.1f}%"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2

        (text_width, text_height), _ = cv2.getTextSize(
            avg_text, font, font_scale, thickness
        )

        padding = 12

        x1 = frame.shape[1] - text_width - (2 * padding) - 10
        y1 = 10
        x2 = frame.shape[1] - 10
        y2 = y1 + text_height + (2 * padding)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), -1)

        cv2.putText(frame,
                    avg_text,
                    (x1 + padding, y2 - padding),
                    font,
                    font_scale,
                    (0, 255, 255),
                    thickness)

        cv2.imshow("Emotion Detector", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()