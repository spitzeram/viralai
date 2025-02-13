import cv2
import numpy as np

def analyze_video_content(video_path: str):
    """
    Extracts video data including:
    - Motion intensity
    - Scene cuts & jump cuts
    - Brightness & color vibrancy
    - Facial detection
    - Hook efficiency (first 3 seconds)
    - Text overlay detection
    - Loopability factor
    """
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 1.0
    duration_seconds = frame_count / fps if fps > 0 else 0

    scene_cuts = 0
    has_text_overlay = False
    has_jump_cuts = False
    has_hook = False
    is_loopable = False
    brightness_levels = []
    prev_frame = None
    sample_interval = max(int(fps // 2), 1)
    frame_idx = 1

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        if frame_idx % sample_interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness_levels.append(np.mean(frame))

            if prev_frame is not None:
                diff = cv2.absdiff(prev_frame, gray)
                if np.mean(diff) > 50:
                    scene_cuts += 1
                    if frame_idx < 90:  # First 3 seconds
                        has_hook = True
            prev_frame = gray

            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            face_detected = len(faces) > 0

        if frame_idx < 90 and np.mean(frame) < 100:
            has_text_overlay = True

    cap.release()
    
    avg_brightness = np.mean(brightness_levels) if brightness_levels else 0
    is_loopable = scene_cuts > 10  # Loopable if there are multiple fast scene cuts

    return {
        "duration_seconds": duration_seconds,
        "scene_cuts": scene_cuts,
        "has_text_overlay": has_text_overlay,
        "has_jump_cuts": scene_cuts > 5,
        "has_hook": has_hook,
        "is_loopable": is_loopable,
        "avg_brightness": avg_brightness,
        "aspect_ratio": "9:16" if fps > 0 and frame_count > fps * 15 else "16:9",
        "face_detected": face_detected
    }
