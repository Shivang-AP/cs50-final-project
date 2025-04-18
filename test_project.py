import numpy as np
import cv2
from project import get_music_for_emotion, detect_faces

def test_get_music_for_emotion():
    assert "Rock - Linkin Park" in get_music_for_emotion("Angry")
    assert get_music_for_emotion("Unknown") == []

def test_detect_faces_blank_image():
    blank = np.zeros((500, 500), dtype="uint8")
    faces = detect_faces(blank)
    assert isinstance(faces, (list, np.ndarray))
    assert len(faces) == 0
 
 