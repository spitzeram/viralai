import numpy as np

def compute_viral_score(video_data, audio_data):
    """
    Determines the viral score based on:
    - Video format, motion, scene cuts
    - Speech clarity, background noise, music
    - Hook efficiency, engagement potential
    """
    duration = video_data.get('duration_seconds', 0)
    motion = video_data.get('scene_cuts', 0)
    aspect_ratio = video_data.get('aspect_ratio', "Unknown")
    face_detected = video_data.get('face_detected', False)
    has_text_overlay = video_data.get('has_text_overlay', False)
    has_hook = video_data.get('has_hook', False)
    is_loopable = video_data.get('is_loopable', False)
    avg_brightness = video_data.get('avg_brightness', 0)

    speech_text = audio_data.get('speech_text', "")
    loudness = audio_data.get('loudness', 0)
    background_noise_level = audio_data.get('background_noise_level', 0)

    score = 0
    reasons = []
    improvement_tips = []
    platform_scores = {"TikTok": 0, "Instagram": 0, "YouTube Shorts": 0}

    if aspect_ratio == "9:16":
        score += 20
        platform_scores["TikTok"] += 20
        platform_scores["Instagram"] += 15
        reasons.append("Optimized vertical format for TikTok and Reels.")
    elif aspect_ratio == "16:9":
        score += 10
        platform_scores["YouTube Shorts"] += 15
        reasons.append("Standard format, more suitable for YouTube.")

    if has_hook:
        score += 15
        platform_scores["TikTok"] += 20
        reasons.append("Strong hook detected in the first 3 seconds.")
    else:
        improvement_tips.append("Start with an eye-catching moment or surprising statement.")
    
    if is_loopable:
        score += 15
        platform_scores["TikTok"] += 20
        reasons.append("Loopable video detected. Higher chance of replays.")
    else:
        improvement_tips.append("Try making the video loop seamlessly at the end.")
    
    if has_text_overlay:
        score += 10
        platform_scores["TikTok"] += 10
        reasons.append("Text overlays/subtitles detected. Improves engagement.")
    else:
        improvement_tips.append("Add captions or bold key phrases to increase watch time.")
    
    if loudness > -10:
        score += 15
        reasons.append("Good audio clarity and volume.")
    else:
        improvement_tips.append("Increase audio clarity. Reduce background noise if present.")
    
    score = min(score, 100)
    max_platform = max(platform_scores, key=platform_scores.get)
    
    return {
        "viral_score": score,
        "platform_recommendation": max_platform,
        "reasons": reasons,
        "platform_scores": platform_scores,
        "improvement_tips": improvement_tips
    }