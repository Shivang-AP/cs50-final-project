import numpy as np
import cv2
from tensorflow.keras.models import load_model

# ✅ Load trained model (No retraining needed)
model = load_model("emotion_model.h5")

# ✅ Load OpenCV Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ✅ Define emotion labels
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# ✅ Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6)  # Face detection

    if len(faces) == 0:
        cv2.putText(frame, "No Face Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    else:
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]  # Extract face
            roi_gray = cv2.resize(roi_gray, (48, 48))  # Resize to model input
            roi_gray = roi_gray.astype("float") / 255.0  # Normalize
            roi_gray = np.expand_dims(roi_gray, axis=0)
            roi_gray = np.expand_dims(roi_gray, axis=-1)

            # ✅ Predict emotion
            prediction = model.predict(roi_gray)[0]
            label = emotion_labels[np.argmax(prediction)]

            # ✅ Draw rectangle & label
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Emotion Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()
