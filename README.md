# Emotion-Based Music Recommender 🎵🙂

#### Video Demo: <Create a README.md text file (named exactly that!) in your ~/project folder that explains your project. This file should include your Project title, the URL of your video (created in step 1 above) and a description of your project. You may use the below as a template.

    # YOUR PROJECT TITLE
    #### Video Demo:  <https://youtu.be/LoJ-vQrtcEc>
    #### Description:
    TODO

#### Description:

This project is a web-based emotion recognition app that recommends music based on the user's facial expression.

It uses:
- **Flask** for the web framework
- **OpenCV** to access and process the webcam feed
- **TensorFlow/Keras** for emotion classification via a pre-trained deep learning model
- **HTML + JavaScript** on the frontend for webcam capture and AJAX requests

### How it Works

1. User accesses the web app through the homepage (`/`).
2. The webcam feed is displayed and the user can click “Capture”.
3. The frontend sends the captured frame to the server in Base64 format.
4. The backend:
   - Decodes the image
   - Detects faces
   - Predicts the emotion using the ML model
   - Matches the emotion with music recommendations
5. A JSON response is returned with the detected emotion and music list.

### Files

- `project.py`: Main backend file. Contains Flask routes, face detection, and emotion classification logic.
- `test_project.py`: Tests for key functions like emotion-to-music mapping and face detection.
- `templates/index.html`: Frontend HTML with video stream and capture logic (not included here).
- `requirements.txt`: Lists the Python dependencies.

### Design Choices

- I chose to separate model inference and recommendation logic into separate functions to improve testability.
- To simplify the testing setup, the emotion model is only used in the main script; tests avoid direct model predictions.

### Future Improvements

- Add more refined music categories using a Spotify API.
- Improve model accuracy with a larger dataset.
- Make the UI more interactive and mobile-friendly.

---
## NOTE - THIS PROJECT REQUIRES A HUGE DATASET WHICH I HAD TO DELETE BECAUSE OF SUBMISIION ISSUES 
## ANY RADOM DATA OF FACIAL EXPRESSION CAN BE USED IN THIS 
# SORRY FOR LAST MOMENT CHANGE
Made with determination for CS50P
