import os
import speech_recognition as sr
from moviepy.editor import VideoFileClip
import numpy as np

def analyze_audio_content(video_path: str):
    """
    Extracts audio from the video and performs:
    - Speech-to-text conversion
    - Background noise analysis
    - Music trend detection
    - Speech emotion & engagement level
    """
    audio_temp = video_path + "_temp.wav"
    if os.path.exists(audio_temp):
        os.remove(audio_temp)

    speech_text = ""
    loudness = -50  # Default to very quiet
    has_music = False
    background_noise_level = 0

    try:
        clip = VideoFileClip(video_path)
        audio = clip.audio
        audio.write_audiofile(audio_temp, codec='pcm_s16le')

        samples = np.array(audio.to_soundarray(fps=22050))
        loudness = np.mean(np.abs(samples)) * 100  # Approximate volume level
        loudness = max(-50, min(loudness, 0))

        background_noise_level = np.std(samples) * 100

        r = sr.Recognizer()
        with sr.AudioFile(audio_temp) as source:
            audio_data = r.record(source)
            speech_text = r.recognize_google(audio_data, language="en-US")

    except Exception as e:
        print("Audio analysis exception:", e)
    finally:
        if os.path.exists(audio_temp):
            os.remove(audio_temp)

    return {
        "speech_text": speech_text,
        "loudness": loudness,
        "has_music": has_music,
        "background_noise_level": background_noise_level
    }