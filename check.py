import cv2
from deepface import DeepFace
import random
from flask import Flask, render_template, Response, jsonify

app = Flask(__name__)

def get_emotion(frame):
    try:
        analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        if analysis and len(analysis) > 0:
            emotions = analysis[0]['emotion']
            dominant_emotion = max(emotions, key=emotions.get)  # Get the most confident emotion
            return dominant_emotion
        else:
            return "neutral"  # Default if no emotion is detected
    except Exception as e:
        print(f"Error detecting emotion: {e}")
        return "neutral"  # Return neutral instead of failing


def recommend_music(emotion):
    music_dict = {
        'happy': ['Happy - Pharrell Williams', 'Can’t Stop the Feeling - Justin Timberlake', 'Uptown Funk - Bruno Mars'],
        'sad': ['Someone Like You - Adele', 'Fix You - Coldplay', 'Yesterday - The Beatles'],
        'angry': ['Smells Like Teen Spirit - Nirvana', 'Break Stuff - Limp Bizkit', 'In The End - Linkin Park'],
        'surprise': ['Bohemian Rhapsody - Queen', 'Thriller - Michael Jackson', 'Take On Me - A-ha'],
        'fear': ['Lose Yourself - Eminem', 'Boulevard of Broken Dreams - Green Day', 'Creep - Radiohead'],
        'disgust': ['Before He Cheats - Carrie Underwood', 'Irreplaceable - Beyoncé', 'Rolling in the Deep - Adele'],
        'neutral': ['Imagine - John Lennon', 'Somewhere Only We Know - Keane', 'Let it Be - The Beatles']
    }
    return music_dict.get(emotion, ['No recommendations available'])

def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detect_emotion')
@app.route('/detect_emotion')
def detect_emotion():
    cap = cv2.VideoCapture(0)
    success, frame = cap.read()
    cap.release()

    if not success:
        return jsonify({"error": "Could not capture image"}), 400

    # Convert BGR to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    emotion = get_emotion(frame)
    recommendations = recommend_music(emotion)
    
    return jsonify({"emotion": emotion, "recommendations": recommendations})


if __name__ == "__main__":
    app.run(debug=True)
