from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import base64

app = Flask(__name__)

# Load trained model
try:
    model = load_model("emotion_model.h5")
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
except Exception as e:
    print(f"Error loading model: {e}")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

music_recommendations = {
    "Angry": ["Rock - Linkin Park", "Metal - Metallica", "Rap - Eminem"],
    "Disgust": ["Calm - Lo-Fi", "Piano - Yiruma", "Acoustic - Ed Sheeran"],
    "Fear": ["Chill - Coldplay", "Instrumental - Hans Zimmer"],
    "Happy": ["Pop - Dua Lipa", "Dance - Avicii", "EDM - Marshmello"],
    "Neutral": ["Indie - Arctic Monkeys", "Jazz - Miles Davis"],
    "Sad": ["Acoustic - Taylor Swift", "Soft Rock - The Beatles"],
    "Surprise": ["Electronic - Daft Punk", "Alternative - Imagine Dragons"]
}

def detect_faces(gray_image):
    return face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=6)

def predict_emotion(gray_image):
    faces = detect_faces(gray_image)
    if len(faces) == 0:
        return "No Face Detected"

    for (x, y, w, h) in faces:
        roi_gray = gray_image[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi_gray = roi_gray.astype("float") / 255.0
        roi_gray = np.expand_dims(roi_gray, axis=0)
        roi_gray = np.expand_dims(roi_gray, axis=-1)
        prediction = model.predict(roi_gray)[0]
        return emotion_labels[np.argmax(prediction)]

def get_music_for_emotion(emotion):
    return music_recommendations.get(emotion, [])

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/capture', methods=['POST'])
def capture():
    data = request.json['image']
    image_data = base64.b64decode(data.split(',')[1])
    nparr = np.frombuffer(image_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    emotion = predict_emotion(gray)
    music = get_music_for_emotion(emotion) if emotion != "No Face Detected" else []

    return jsonify({"emotion": emotion, "music": music})

if __name__ == '__main__':
    app.run(debug=True)
