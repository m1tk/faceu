import cv2
import numpy as np
import math

cfg = {
    # Sharpness/Blur
    'sharpness_sigma': 0.8,
    'laplacian_ideal': 200.0,     # Stricter: Higher is better
    'tenengrad_ideal': 1_500_000.0, # Stricter: Higher is better

    # Pose
    'max_acceptable_pitch_abs': 15.0, # Stricter: degrees
    'max_acceptable_yaw_abs': 15.0,   # Stricter: degrees
    'max_acceptable_roll_abs': 10.0,  # Stricter: degrees
    'pose_pitch_weight': 0.35,
    'pose_yaw_weight': 0.35,
    'pose_roll_weight': 0.15,

    # Illumination
    'brightness_ideal_low': 90.0,   # Slightly stricter
    'brightness_ideal_high': 180.0,
    'min_acceptable_contrast_std': 30.0, # Stricter

    # Occlusion (Eyes and Mouth)
    'ear_threshold': 0.28, # Stricter: Eyes more open
    'mar_threshold': 0.45,  # Stricter: Mouth more closed

    # Resolution
    'min_face_size_pixels': 120, # Stricter: Minimum width/height for ROI

    # Overall Weights for combined score (sum should be 1.0)
    "weights": {
        "sharpness": 0.35, # More important
        "pose": 0.25,
        "brightness": 0.10,
        "contrast": 0.10,
        "eyes_open": 0.10,
        "mouth_closed": 0.05,
        "resolution": 0.15 # More important
    },
    # Thresholds for 'reason' messages (score below this triggers a reason)
    "reason_thresholds": {
        "sharpness": 0.7, # Stricter
        "pose": 0.7,      # Stricter
        "brightness": 0.5,
        "contrast": 0.5,
        "eyes_open": 1.0,
        "mouth_closed": 1.0,
        "resolution": 0.8 # Stricter
    }
}

def calculate_laplacian_variance(image_gray, ksize=3, sigma=0.5):
    blurred_image = cv2.GaussianBlur(image_gray, (0,0), sigma)
    laplacian_variance = cv2.Laplacian(blurred_image, cv2.CV_64F, ksize=ksize).var()
    return laplacian_variance

def calculate_tenengrad(image_gray):
    sobelx = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=3)
    tenengrad_score = (sobelx**2 + sobely**2).sum()
    return tenengrad_score

def estimate_pose(landmarks_points, img_h, img_w):
    # This function expects 'landmarks_points' to be a dictionary where keys are meaningful
    # names (e.g., 'nose_tip', 'chin') and values are (x,y) tuples.
    # It attempts to map face_recognition's output to these specific points.

    # 3D model points (fixed)
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (-225.0, 170.0, -135.0),     # Left eye left corner
        (225.0, 170.0, -135.0),      # Right eye right corner
        (-150.0, -150.0, -125.0),    # Left mouth corner
        (150.0, -150.0, -125.0)      # Right mouth corner
    ])

    # Reconstruct the 6 key points from the provided 'landmarks_points' dictionary
    try:
        image_points = np.array([
            landmarks_points['nose_tip'],
            landmarks_points['chin'],
            landmarks_points['left_eye_left_corner'],
            landmarks_points['right_eye_right_corner'],
            landmarks_points['left_mouth_corner'],
            landmarks_points['right_mouth_corner']
        ], dtype="double")
    except KeyError as e:
        return 900.0, 900.0, 900.0 # Large values to signify failure

    # Camera internals (approximated for general use within the ROI)
    focal_length = img_w # A common heuristic
    center = (img_w / 2, img_h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")

    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion

    success, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

    if not success:
        return 900.0, 900.0, 900.0 # Indicate failure to solve PnP

    rmat, _ = cv2.Rodrigues(rotation_vector)
    
    sy = math.sqrt(rmat[0,0] * rmat[0,0] + rmat[1,0] * rmat[1,0])
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(rmat[2,1] , rmat[2,2]) # Pitch
        y = math.atan2(-rmat[2,0], sy)       # Yaw
        z = math.atan2(rmat[1,0], rmat[0,0]) # Roll
    else:
        x = math.atan2(-rmat[1,2], rmat[1,1]) # Pitch
        y = math.atan2(-rmat[2,0], sy)       # Yaw
        z = 0                                # Roll
    
    pitch = math.degrees(x)
    yaw = math.degrees(y)
    roll = math.degrees(z)

    return pitch, yaw, roll

def calculate_brightness(face_roi_gray):
    return np.mean(face_roi_gray)

def calculate_contrast(face_roi_gray):
    return np.std(face_roi_gray)

def eye_aspect_ratio(eye_landmarks_list):
    if len(eye_landmarks_list) != 6: return 0.0
    eye_landmarks = np.array(eye_landmarks_list)

    A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
    
    if C == 0: return 0.0
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(mouth_landmarks_list):
    if len(mouth_landmarks_list) != 12: return 0.0
    mouth_landmarks = np.array(mouth_landmarks_list)

    A = np.linalg.norm(mouth_landmarks[1] - mouth_landmarks[7])
    B = np.linalg.norm(mouth_landmarks[2] - mouth_landmarks[6])
    C = np.linalg.norm(mouth_landmarks[3] - mouth_landmarks[5])
    
    D = np.linalg.norm(mouth_landmarks[0] - mouth_landmarks[4])

    if D == 0: return 0.0
    mar = (A + B + C) / (3.0 * D)
    return mar

def assess_face_quality(face_roi_rgb, landmarks_list):
    """
    Assesses the quality of a detected face ROI for recognition.

    Args:
        face_roi_rgb (np.array): The cropped face image (RGB or BGR, will convert to grayscale).
        config (dict, optional): Dictionary for fine-tuning parameters.
                                 If None, uses default values.

    Returns:
        dict: A dictionary containing various quality scores and an overall quality assessment.
    """
    landmarks = landmarks_list[0] if landmarks_list else {}

    face_roi_gray = cv2.cvtColor(face_roi_rgb, cv2.COLOR_RGB2GRAY) # Assuming RGB input

    img_h, img_w = face_roi_rgb.shape[:2]
    
    # 1. Sharpness/Blur Assessment
    laplacian_var = calculate_laplacian_variance(face_roi_gray, sigma=cfg['sharpness_sigma'])
    tenengrad_score = calculate_tenengrad(face_roi_gray)
    
    normalized_laplacian = min(laplacian_var / cfg['laplacian_ideal'], 1.0)
    normalized_tenengrad = min(tenengrad_score / cfg['tenengrad_ideal'], 1.0)
    sharpness_score = (normalized_laplacian + normalized_tenengrad) / 2.0

    # 2. Pose Estimation
    # Mapping face_recognition's output to the 6 specific points estimate_pose expects.
    pose_landmarks_map = {}
    if 'nose_bridge' in landmarks and len(landmarks['nose_bridge']) > 2: # nose tip usually at index 2 or 3 of nose_bridge or nose_tip if separate
        pose_landmarks_map['nose_tip'] = landmarks['nose_bridge'][2]
    elif 'nose_tip' in landmarks and len(landmarks['nose_tip']) > 0: # Some models might have a dedicated 'nose_tip'
         pose_landmarks_map['nose_tip'] = landmarks['nose_tip'][0]
    
    if 'chin' in landmarks and len(landmarks['chin']) > 8: # Chin is usually at the bottom-most point
        pose_landmarks_map['chin'] = landmarks['chin'][8]
    
    if 'left_eye' in landmarks and len(landmarks['left_eye']) > 0:
        pose_landmarks_map['left_eye_left_corner'] = landmarks['left_eye'][0]
    if 'right_eye' in landmarks and len(landmarks['right_eye']) > 3:
        pose_landmarks_map['right_eye_right_corner'] = landmarks['right_eye'][3]
    
    if 'mouth' in landmarks and len(landmarks['mouth']) > 0:
        pose_landmarks_map['left_mouth_corner'] = landmarks['mouth'][0]
    if 'mouth' in landmarks and len(landmarks['mouth']) > 6:
        pose_landmarks_map['right_mouth_corner'] = landmarks['mouth'][6]

    pitch, yaw, roll = estimate_pose(pose_landmarks_map, img_h, img_w)

    # Check if estimate_pose failed (returned extreme values)
    if pitch == 900.0 and yaw == 900.0 and roll == 900.0:
        pose_score = 0.0 # Indicate very poor or uncalculable pose
    else:
        score_pitch = max(0.0, 1.0 - abs(pitch) / cfg['max_acceptable_pitch_abs'])
        score_yaw = max(0.0, 1.0 - abs(yaw) / cfg['max_acceptable_yaw_abs'])
        score_roll = max(0.0, 1.0 - abs(roll) / cfg['max_acceptable_roll_abs'])
        pose_score = (score_pitch * cfg['pose_pitch_weight'] + 
                      score_yaw * cfg['pose_yaw_weight'] + 
                      score_roll * cfg['pose_roll_weight'])
        pose_score = max(0.0, min(pose_score, 1.0)) # Clamp between 0 and 1

    # 3. Illumination Assessment
    brightness = calculate_brightness(face_roi_gray)
    contrast = calculate_contrast(face_roi_gray)

    if cfg['brightness_ideal_low'] <= brightness <= cfg['brightness_ideal_high']:
        brightness_score = 1.0
    elif brightness < cfg['brightness_ideal_low']:
        brightness_score = brightness / cfg['brightness_ideal_low']
    else: # brightness > BRIGHTNESS_IDEAL_HIGH
        brightness_score = (255 - brightness) / (255 - cfg['brightness_ideal_high'])
    brightness_score = max(0.0, min(brightness_score, 1.0))

    contrast_score = min(contrast / cfg['min_acceptable_contrast_std'], 1.0)
    
    # 4. Occlusion (Eyes and Mouth)
    left_eye_points = landmarks.get('left_eye', [])
    right_eye_points = landmarks.get('right_eye', [])
    mouth_points = landmarks.get('mouth', [])

    ear_left = eye_aspect_ratio(left_eye_points)
    ear_right = eye_aspect_ratio(right_eye_points)
    
    if ear_left > cfg['ear_threshold'] and ear_right > cfg['ear_threshold']:
        eyes_open_score = 1.0
    elif ear_left > cfg['ear_threshold'] or ear_right > cfg['ear_threshold']:
        eyes_open_score = 0.5 # One eye open, customize this penalty
    else:
        eyes_open_score = 0.0

    mar = mouth_aspect_ratio(mouth_points)
    mouth_closed_score = 1.0 if mar < cfg['mar_threshold'] else 0.0

    # 5. Resolution/Pixel Density
    face_width = img_w
    face_height = img_h
    
    if face_width < cfg['min_face_size_pixels'] or face_height < cfg['min_face_size_pixels']:
        resolution_score = 0.0
    else:
        resolution_score = min((face_width / cfg['min_face_size_pixels'] + face_height / cfg['min_face_size_pixels']) / 2.0, 1.0)

    # Combine scores into an overall quality score (weighted sum)
    overall_quality_score = (
        sharpness_score * cfg["weights"]["sharpness"] +
        pose_score * cfg["weights"]["pose"] +
        brightness_score * cfg["weights"]["brightness"] +
        contrast_score * cfg["weights"]["contrast"] +
        eyes_open_score * cfg["weights"]["eyes_open"] +
        mouth_closed_score * cfg["weights"]["mouth_closed"] +
        resolution_score * cfg["weights"]["resolution"]
    )
    return overall_quality_score