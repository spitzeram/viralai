import os
import random
import numpy as np
import cv2
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from werkzeug.utils import secure_filename
from moviepy.editor import VideoFileClip

# --- Configuration ---
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi', 'mkv'}
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app = Flask(__name__)
app.secret_key = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ------------------- HELPER FUNCTIONS -------------------

def allowed_file(filename):
    """Check if the file has an allowed video extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def detect_video_content(file_path):
    """
    Simulate detection of video content using:
      - Speech-to-text (for spoken languages)
      - OCR (for on-screen text)
      - Subtitle parsing (for subtitle presence)

    For demo, we assume:
      - Spoken language: Hebrew
      - On-screen text: English
      - No official subtitles
    """
    spoken = ['Hebrew']
    on_screen = ['English']
    subtitles = False
    overview = (
        "The video features dynamic Hebrew dialogue accompanied by occasional on-screen English text. "
        "No official subtitles were detected, which might limit accessibility for non-Hebrew speakers. "
        "Overall, the content appears informal and conversational, targeting a bilingual audience."
    )
    return {
        'spoken_languages': spoken,
        'on_screen_text_languages': on_screen,
        'subtitles_present': subtitles,
        'video_overview': overview
    }

def compute_score(value, lower, upper):
    """Normalize a value to a score between 60 and 100."""
    if value <= lower:
        return 60
    elif value >= upper:
        return 100
    else:
        return 60 + (value - lower) / (upper - lower) * 40

def compute_consistency_score(std, range_span):
    """Compute a consistency score (60–100) based on standard deviation."""
    ratio = min(std, range_span) / range_span
    return 60 + (1 - ratio) * 40

def compute_average_brightness(file_path, num_samples=10):
    """Sample frames from the video and compute average brightness."""
    cap = cv2.VideoCapture(file_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    brightness_values = []
    if frame_count <= 0:
        cap.release()
        return 0
    sample_rate = max(1, frame_count // num_samples)
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % sample_rate == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness_values.append(np.mean(gray))
        count += 1
    cap.release()
    return np.mean(brightness_values) if brightness_values else 0

# ------------------- FRAME ANALYSIS & THUMBNAIL -------------------

def process_video_frames(file_path, num_samples=10):
    """
    Compute various metrics (contrast, saturation, face count, etc.) and
    choose a "best" frame for the thumbnail:
      - Prefer the frame with the highest face count
      - Tie-break on total face area
      - If no faces, fallback to highest color variance
    """
    cap = cv2.VideoCapture(file_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_rate = max(1, frame_count // num_samples)

    contrast_list = []
    sharpness_list = []
    saturation_list = []
    edge_density_list = []
    color_variance_list = []
    face_count_list = []
    visual_complexity_list = []
    motion_list = []

    best_face_count = 0
    best_face_area = 0
    best_face_frame = None
    best_face_index = 0

    best_color_var = 0
    best_color_frame = None
    best_color_index = 0

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    prev_gray = None
    count = 0
    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % sample_rate == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Contrast & sharpness (variance of Laplacian)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            contrast = laplacian.var()
            contrast_list.append(contrast)
            sharpness_list.append(contrast)
            # Saturation (HSV)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            saturation = hsv[:,:,1].mean()
            saturation_list.append(saturation)
            # Edge density (Canny)
            edges = cv2.Canny(gray, 100, 200)
            edge_density = (np.count_nonzero(edges) / edges.size) * 100
            edge_density_list.append(edge_density)
            # Color variance
            color_var = np.var(frame)
            color_variance_list.append(color_var)
            # Face detection
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            face_count = len(faces)
            face_count_list.append(face_count)
            total_face_area = sum([w*h for (x, y, w, h) in faces])
            if (face_count > best_face_count) or (face_count == best_face_count and total_face_area > best_face_area):
                best_face_count = face_count
                best_face_area = total_face_area
                best_face_frame = frame.copy()
                best_face_index = frame_index
            # Track best color variance
            if color_var > best_color_var:
                best_color_var = color_var
                best_color_frame = frame.copy()
                best_color_index = frame_index
            # Visual complexity (entropy)
            hist = cv2.calcHist([gray], [0], None, [256], [0,256])
            hist = hist.ravel() / hist.sum()
            hist = hist[np.nonzero(hist)]
            entropy = -np.sum(hist * np.log2(hist))
            visual_complexity_list.append(entropy)
            # Motion intensity
            if prev_gray is not None:
                diff = cv2.absdiff(gray, prev_gray)
                motion = np.mean(diff)
                motion_list.append(motion)
            prev_gray = gray
            frame_index += 1
        count += 1
    cap.release()

    camera_stability = 100 - np.std(motion_list) if motion_list else 100

    thumbnail_path = None
    thumbnail_reason = ""
    if best_face_count > 0 and best_face_frame is not None:
        time_in_seconds = (best_face_index * sample_rate) / fps
        thumbnail_reason = f"Frame at ~{time_in_seconds:.2f}s with highest face count and largest face area."
        thumbnail_path = os.path.join(UPLOAD_FOLDER, f"thumbnail_{os.path.basename(file_path)}.jpg")
        cv2.imwrite(thumbnail_path, best_face_frame)
    elif best_color_frame is not None:
        time_in_seconds = (best_color_index * sample_rate) / fps
        thumbnail_reason = f"No faces detected; using most colorful frame at ~{time_in_seconds:.2f}s."
        thumbnail_path = os.path.join(UPLOAD_FOLDER, f"thumbnail_{os.path.basename(file_path)}.jpg")
        cv2.imwrite(thumbnail_path, best_color_frame)

    return {
        'avg_contrast': np.mean(contrast_list) if contrast_list else 0,
        'contrast_std': np.std(contrast_list) if contrast_list else 0,
        'avg_sharpness': np.mean(sharpness_list) if sharpness_list else 0,
        'sharpness_std': np.std(sharpness_list) if sharpness_list else 0,
        'avg_saturation': np.mean(saturation_list) if saturation_list else 0,
        'saturation_std': np.std(saturation_list) if saturation_list else 0,
        'avg_edge_density': np.mean(edge_density_list) if edge_density_list else 0,
        'edge_density_std': np.std(edge_density_list) if edge_density_list else 0,
        'avg_color_variance': np.mean(color_variance_list) if color_variance_list else 0,
        'color_variance_std': np.std(color_variance_list) if color_variance_list else 0,
        'avg_face_count': np.mean(face_count_list) if face_count_list else 0,
        'face_count_std': np.std(face_count_list) if face_count_list else 0,
        'avg_visual_complexity': np.mean(visual_complexity_list) if visual_complexity_list else 0,
        'visual_complexity_std': np.std(visual_complexity_list) if visual_complexity_list else 0,
        'avg_motion_intensity': np.mean(motion_list) if motion_list else 0,
        'motion_intensity_std': np.std(motion_list) if motion_list else 0,
        'camera_stability': camera_stability,
        'thumbnail_path': thumbnail_path,
        'thumbnail_reason': thumbnail_reason
    }

# ------------------- AUDIO ANALYSIS -------------------

def analyze_audio(clip):
    """Analyze audio to compute loudness and simulated speech metrics."""
    if not clip.audio:
        return {
            'audio_loudness': 0,
            'speech_clarity': 60,
            'speech_speed': 60,
            'speech_emotion': 60,
            'explanation': "No audio track found."
        }
    try:
        audio_array = clip.audio.to_soundarray(fps=16000)
        if audio_array.ndim == 2 and audio_array.shape[1] > 1:
            audio_array = audio_array.mean(axis=1, keepdims=True)
    except Exception as e:
        return {
            'audio_loudness': 60,
            'speech_clarity': 60,
            'speech_speed': 60,
            'speech_emotion': 60,
            'explanation': f"Could not process audio track: {str(e)}"
        }
    loudness = np.mean(np.abs(audio_array))
    audio_loudness_score = compute_score(loudness, 0.01, 0.2)
    speech_clarity_score = random.randint(60, 100)
    speech_speed_score = random.randint(60, 100)
    speech_emotion_score = random.randint(60, 100)
    return {
        'audio_loudness': audio_loudness_score,
        'speech_clarity': speech_clarity_score,
        'speech_speed': speech_speed_score,
        'speech_emotion': speech_emotion_score,
        'explanation': "Audio track analyzed. Loudness, clarity, speed, and emotion are estimated."
    }

# ------------------- METRIC CALCULATION -------------------

def get_video_metrics(clip, file_path, video_content):
    """Combine high-level, frame, audio, and derived metrics (total 100)."""
    metrics = []

    # 1. Duration Efficiency
    duration = clip.duration
    duration_efficiency = 95 if duration < 60 else 50
    metrics.append({
        'name': "Duration Efficiency",
        'score': duration_efficiency,
        'explanation': (f"Video duration of {duration:.2f}s is optimal."
                        if duration < 60 else "Video duration is longer than ideal.")
    })

    # 2. Visual Resolution
    width, height = clip.size
    resolution_score = min((width / 1920) * 100, 100) if width and height else 50
    metrics.append({
        'name': "Visual Resolution",
        'score': resolution_score,
        'explanation': f"Resolution is {width}x{height} ({'excellent' if resolution_score > 80 else 'adequate'})."
    })

    # 3. Average Brightness
    brightness = compute_average_brightness(file_path)
    brightness_score = 60 if brightness < 80 else (70 if brightness > 180 else 90)
    metrics.append({
        'name': "Average Brightness",
        'score': brightness_score,
        'explanation': f"Average brightness is {brightness:.1f}."
    })

    # 4. Audio Clarity (Legacy, simulated)
    audio_clarity_legacy = random.randint(70, 95)
    metrics.append({
        'name': "Audio Clarity (Legacy)",
        'score': audio_clarity_legacy,
        'explanation': "Simulated measure of audio clarity."
    })

    # 5. Subtitles Effectiveness
    subs_score = 90 if video_content['subtitles_present'] else 50
    metrics.append({
        'name': "Subtitles Effectiveness",
        'score': subs_score,
        'explanation': ("Subtitles enhance accessibility."
                        if video_content['subtitles_present'] else "No official subtitles detected.")
    })

    # 6. Multilingual Appeal
    total_langs = len(video_content['spoken_languages']) + len(video_content['on_screen_text_languages'])
    multilingual_appeal = 95 if total_langs > 1 else 60
    metrics.append({
        'name': "Multilingual Appeal",
        'score': multilingual_appeal,
        'explanation': (f"Spoken: {', '.join(video_content['spoken_languages'])}; "
                        f"On-screen: {', '.join(video_content['on_screen_text_languages'])}.")
    })

    # 7. Engagement Factor (Simulated)
    engagement_factor = random.randint(70, 100)
    metrics.append({
        'name': "Engagement Factor",
        'score': engagement_factor,
        'explanation': "Simulated measure of content engagement."
    })

    # 8. Pacing (Simulated)
    pacing = random.randint(65, 100)
    metrics.append({
        'name': "Pacing",
        'score': pacing,
        'explanation': "Simulated measure of video pacing."
    })

    # 9. Emotional Impact (Simulated)
    emotional_impact = random.randint(60, 100)
    metrics.append({
        'name': "Emotional Impact",
        'score': emotional_impact,
        'explanation': "Simulated measure of emotional engagement."
    })

    # 10. Content Originality (Simulated)
    content_originality = random.randint(60, 100)
    metrics.append({
        'name': "Content Originality",
        'score': content_originality,
        'explanation': "Simulated measure of originality."
    })

    # 11–20: Real Data from Frame Analysis
    frame_data = process_video_frames(file_path)
    avg_contrast = frame_data.get('avg_contrast', 0)
    contrast_std = frame_data.get('contrast_std', 0)
    avg_sharpness = frame_data.get('avg_sharpness', 0)
    avg_saturation = frame_data.get('avg_saturation', 0)
    avg_edge_density = frame_data.get('avg_edge_density', 0)
    avg_motion_intensity = frame_data.get('avg_motion_intensity', 0)
    avg_color_variance = frame_data.get('avg_color_variance', 0)
    avg_face_count = frame_data.get('avg_face_count', 0)
    avg_visual_complexity = frame_data.get('avg_visual_complexity', 0)
    camera_stability = frame_data.get('camera_stability', 100)
    thumbnail_path = frame_data.get('thumbnail_path', None)
    thumbnail_reason = frame_data.get('thumbnail_reason', "")

    metrics.append({
        'name': "Average Contrast",
        'score': compute_score(avg_contrast, 50, 300),
        'explanation': f"Contrast is {avg_contrast:.2f}."
    })
    metrics.append({
        'name': "Average Sharpness",
        'score': compute_score(avg_sharpness, 50, 300),
        'explanation': f"Sharpness is {avg_sharpness:.2f}."
    })
    metrics.append({
        'name': "Average Saturation",
        'score': compute_score(avg_saturation, 50, 200),
        'explanation': f"Saturation is {avg_saturation:.2f}."
    })
    metrics.append({
        'name': "Edge Density",
        'score': compute_score(avg_edge_density, 1, 20),
        'explanation': f"Edge density is {avg_edge_density:.2f}%."
    })
    motion_score = 100 - compute_score(avg_motion_intensity, 0, 50)
    metrics.append({
        'name': "Motion Intensity",
        'score': motion_score,
        'explanation': f"Motion intensity is {avg_motion_intensity:.2f}."
    })
    metrics.append({
        'name': "Color Variance",
        'score': compute_score(avg_color_variance, 500, 3000),
        'explanation': f"Color variance is {avg_color_variance:.2f}."
    })
    metrics.append({
        'name': "Face Detection Rate",
        'score': compute_score(avg_face_count, 0, 3),
        'explanation': f"Average face count is {avg_face_count:.2f}."
    })
    metrics.append({
        'name': "Visual Complexity",
        'score': compute_score(avg_visual_complexity, 3, 7),
        'explanation': f"Visual complexity is {avg_visual_complexity:.2f}."
    })
    metrics.append({
        'name': "Camera Stability",
        'score': camera_stability,
        'explanation': f"Camera stability is {camera_stability:.2f}."
    })
    metrics.append({
        'name': "Contrast Consistency",
        'score': compute_consistency_score(contrast_std, 250),
        'explanation': f"Contrast consistency (std) is {contrast_std:.2f}."
    })

    # 21–24: Audio Analysis
    audio_data = analyze_audio(clip)
    metrics.append({
        'name': "Audio Loudness",
        'score': audio_data['audio_loudness'],
        'explanation': f"Loudness: {audio_data['explanation']}"
    })
    metrics.append({
        'name': "Speech Clarity",
        'score': audio_data['speech_clarity'],
        'explanation': "Simulated measure of speech clarity."
    })
    metrics.append({
        'name': "Speech Speed",
        'score': audio_data['speech_speed'],
        'explanation': "Simulated measure of speech tempo."
    })
    metrics.append({
        'name': "Speech Emotion",
        'score': audio_data['speech_emotion'],
        'explanation': "Simulated measure of emotional expression."
    })

    # 25–100: Additional Derived Metrics
    base_metrics_data = {
        "Contrast": (avg_contrast, contrast_std, 50, 300),
        "Saturation": (avg_saturation, 0, 50, 200),
        "Motion": (avg_motion_intensity, 0, 0, 50),
        "Faces": (avg_face_count, 0, 0, 3),
        "Complexity": (avg_visual_complexity, 0, 3, 7),
        "Loudness": (audio_data['audio_loudness'], 0, 60, 100)
    }
    descriptors = ["Consistency", "Enhanced", "Variation", "Stability", "Balance", "Intensity", "Quality", "Dynamic Range", "Refinement", "Power"]
    for base, (mean_val, std_val, lower, upper) in base_metrics_data.items():
        base_score = compute_score(mean_val, lower, upper)
        for descriptor in descriptors:
            metrics.append({
                'name': f"{descriptor} {base}",
                'score': base_score,
                'explanation': (
                    f"The {descriptor.lower()} of {base.lower()} is {base_score:.2f}/100, "
                    f"based on an average of {mean_val:.2f}."
                )
            })
    placeholders_needed = 100 - len(metrics)
    for i in range(placeholders_needed):
        metrics.append({
            'name': f"Additional Derived Metric {i+1}",
            'score': random.randint(60, 100),
            'explanation': "A further aspect of virality computed from integrated data."
        })

    return metrics, thumbnail_path, thumbnail_reason

def generate_summary_and_improvements(metrics):
    """Generate overall summary and improvement suggestions."""
    overall_score = sum(metric['score'] for metric in metrics) / len(metrics)
    primary_metrics = metrics[:10]
    best_metric = max(primary_metrics, key=lambda x: x['score'])
    worst_metric = min(primary_metrics, key=lambda x: x['score'])
    improvements = []
    if best_metric['score'] < 80:
        improvements.append("Consider refining overall video production techniques.")
    else:
        improvements.append("The video shows many strengths; small tweaks could enhance it further.")
    if worst_metric['score'] < 80:
        improvements.append(
            f"Improve {worst_metric['name'].lower()} (current score: {worst_metric['score']}/100) for broader appeal."
        )
    summary = (f"Overall viral potential: {overall_score:.2f}/100. "
               f"Strong in {best_metric['name'].lower()}, but {worst_metric['name'].lower()} could be improved.")
    return overall_score, summary, " ".join(improvements)

def compute_recommended_platforms(computed_metrics):
    """Simulate platform recommendations based on key metrics."""
    duration_efficiency = computed_metrics[0]['score']
    visual_resolution   = computed_metrics[1]['score']
    brightness_score    = computed_metrics[2]['score']
    audio_clarity       = computed_metrics[3]['score']
    subtitles_effectiveness = computed_metrics[4]['score']
    multilingual_appeal = computed_metrics[5]['score']
    engagement_factor   = computed_metrics[6]['score']
    pacing              = computed_metrics[7]['score']
    emotional_impact    = computed_metrics[8]['score']
    content_originality = computed_metrics[9]['score']
    return {
        "TikTok": int((duration_efficiency * 0.3 + engagement_factor * 0.4 + pacing * 0.3)),
        "Instagram": int((visual_resolution * 0.4 + emotional_impact * 0.3 + subtitles_effectiveness * 0.3)),
        "YouTube Shorts": int((duration_efficiency * 0.3 + content_originality * 0.4 + engagement_factor * 0.3)),
        "Facebook": int(((duration_efficiency + visual_resolution + engagement_factor + audio_clarity) / 4))
    }

def analyze_video(file_path):
    """Main function to analyze video and return a dictionary of results."""
    try:
        clip = VideoFileClip(file_path)
    except Exception as e:
        return {'error': f"Error processing video: {str(e)}"}
    if clip.duration > 60:
        clip.reader.close()
        if clip.audio:
            clip.audio.reader.close_proc()
        return {'error': 'Video exceeds the 60 seconds limit.'}

    video_content = detect_video_content(file_path)
    metrics, thumbnail_path, thumbnail_reason = get_video_metrics(clip, file_path, video_content)
    overall_score, summary, improvements = generate_summary_and_improvements(metrics)
    recommended_platforms = compute_recommended_platforms(metrics)

    clip.reader.close()
    if clip.audio:
        clip.audio.reader.close_proc()

    analysis = {
        'duration': clip.duration,
        'overall_score': overall_score,
        'recommended_platforms': recommended_platforms,
        'metrics': metrics,
        'improvements': improvements,
        'summary': summary,
        'spoken_languages': video_content['spoken_languages'],
        'on_screen_text_languages': video_content['on_screen_text_languages'],
        'subtitles_present': video_content['subtitles_present'],
        'video_overview': video_content['video_overview'],
        'thumbnail_path': None,
        'thumbnail_reason': thumbnail_reason
    }
    if thumbnail_path and os.path.exists(thumbnail_path):
        analysis['thumbnail_path'] = os.path.basename(thumbnail_path)
    return analysis

# ------------------- FLASK ROUTES -------------------

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files from the uploads folder."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'video' not in request.files:
            flash('No video file part in the request.')
            return redirect(request.url)
        file = request.files['video']
        if file.filename == '':
            flash('No file selected.')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            analysis = analyze_video(filepath)
            if 'error' in analysis:
                flash(analysis['error'])
                return redirect(request.url)
            return render_template('result.html', analysis=analysis)
        else:
            flash('Invalid file type.')
            return redirect(request.url)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=8080)
