import cv2
import numpy as np
import onnxruntime as ort
import os
import pickle
import face_recognition
import mediapipe as mp
import logging
import threading
from init import initialize_shared, get_recognizer, reload_embeddings

DATASET_PATH = "dataset"
EMBEDDINGS_FILE = "known_faces_embeddings.pkl"
_retrain_lock = threading.Lock()

logging.basicConfig(level=logging.INFO)

def align_face(face_image_rgb):
    """Aligns a face image based on eye landmarks."""
    landmarks = face_recognition.face_landmarks(face_image_rgb)
    if not landmarks:
        return face_image_rgb, None
    left_eye = np.mean(landmarks[0]['left_eye'], axis=0)
    right_eye = np.mean(landmarks[0]['right_eye'], axis=0)
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))
    eyes_center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)
    M = cv2.getRotationMatrix2D(eyes_center, angle, scale=1.0)
    aligned_face = cv2.warpAffine(face_image_rgb, M, (face_image_rgb.shape[1], face_image_rgb.shape[0]), flags=cv2.INTER_CUBIC)
    
    return aligned_face, landmarks

def preprocess_for_recognizer(face_image_rgb, input_shape, detector_conf_thresh=0.5):
    """
    Uses MediaPipe pose detection to find face landmarks, crops to face ROI,
    then aligns and preprocesses for the recognition model.
    """
    # Initialize MediaPipe pose detection
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=True,  # Use static mode for single images
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=detector_conf_thresh
    )
    
    # Convert RGB to BGR for MediaPipe
    frame_bgr = cv2.cvtColor(face_image_rgb, cv2.COLOR_RGB2BGR)
    
    # Detect pose
    results = pose.process(frame_bgr)
    
    frame_height, frame_width = face_image_rgb.shape[:2]
    face_crop = face_image_rgb  # Default to full image if no pose detected
    
    if results.pose_landmarks:
        # Extract face region from pose landmarks
        landmarks = results.pose_landmarks.landmark
        
        # Get key face landmarks from pose (nose, left/right eye, left/right mouth)
        nose = landmarks[mp_pose.PoseLandmark.NOSE]
        left_eye = landmarks[mp_pose.PoseLandmark.LEFT_EYE]
        right_eye = landmarks[mp_pose.PoseLandmark.RIGHT_EYE]
        left_mouth = landmarks[mp_pose.PoseLandmark.MOUTH_LEFT]
        right_mouth = landmarks[mp_pose.PoseLandmark.MOUTH_RIGHT]
        
        # Convert normalized coordinates to pixel coordinates
        nose_x, nose_y = int(nose.x * frame_width), int(nose.y * frame_height)
        left_eye_x, left_eye_y = int(left_eye.x * frame_width), int(left_eye.y * frame_height)
        right_eye_x, right_eye_y = int(right_eye.x * frame_width), int(right_eye.y * frame_height)
        left_mouth_x, left_mouth_y = int(left_mouth.x * frame_width), int(left_mouth.y * frame_height)
        right_mouth_x, right_mouth_y = int(right_mouth.x * frame_width), int(right_mouth.y * frame_height)
        
        # Calculate face bounding box from facial landmarks
        face_landmarks_x = [nose_x, left_eye_x, right_eye_x, left_mouth_x, right_mouth_x]
        face_landmarks_y = [nose_y, left_eye_y, right_eye_y, left_mouth_y, right_mouth_y]
        
        # Filter out invalid coordinates
        valid_x = [x for x in face_landmarks_x if 0 <= x < frame_width]
        valid_y = [y for y in face_landmarks_y if 0 <= y < frame_height]
        
        if len(valid_x) >= 3 and len(valid_y) >= 3:  # Need at least 3 valid points
            min_x, max_x = min(valid_x), max(valid_x)
            min_y, max_y = min(valid_y), max(valid_y)
            
            # Expand bounding box to include full face with generous padding
            face_width = max_x - min_x
            face_height = max_y - min_y
            
            # Add generous padding to capture forehead, chin, and sides
            padding_x = int(face_width * 0.6)  # 60% padding left/right
            padding_y_top = int(face_height * 0.8)  # 80% padding above for forehead
            padding_y_bottom = int(face_height * 0.6)  # 60% padding below for chin
            
            x1 = max(0, min_x - padding_x)
            y1 = max(0, min_y - padding_y_top)
            x2 = min(frame_width, max_x + padding_x)
            y2 = min(frame_height, max_y + padding_y_bottom)
            
            # Ensure minimum face size
            face_w = x2 - x1
            face_h = y2 - y1
            
            min_face_size = 50  # Same as in face_reco.py
            if face_w >= min_face_size and face_h >= min_face_size:
                # Make bounding box more square for better face recognition
                if face_w > face_h:
                    diff = face_w - face_h
                    y1 = max(0, y1 - diff // 2)
                    y2 = min(frame_height, y2 + diff // 2)
                elif face_h > face_w:
                    diff = face_h - face_w
                    x1 = max(0, x1 - diff // 2)
                    x2 = min(frame_width, x2 + diff // 2)
                
                # Crop to face ROI
                face_crop = face_image_rgb[y1:y2, x1:x2]
                if face_crop.size == 0:
                    face_crop = face_image_rgb
    
    # Clean up MediaPipe resources
    pose.close()
    
    # Align face and prepare for recognition
    aligned, _ = align_face(face_crop)
    face_image = cv2.resize(aligned, input_shape)
    face_image = face_image.astype(np.float32) / 255.0
    face_image = (face_image - 0.5) * 2.0  # Normalize to [-1, 1]
    face_image = np.transpose(face_image, (2, 0, 1))
    return np.expand_dims(face_image, axis=0)

def retrain_and_save_embeddings():
    """
    Scans the dataset directory, generates face embeddings for all images,
    and saves them to a pickle file. This will overwrite the existing embeddings file.
    Only processes users whose face datasets are not already present in saved embeddings.
    Protected by a lock to ensure only one retraining runs at a time.
    """
    with _retrain_lock:
        try:
            initialize_shared("models", EMBEDDINGS_FILE)
            recognizer_session, recognizer_input_name, recognizer_output_name = get_recognizer()
            input_shape = tuple(recognizer_session.get_inputs()[0].shape[-2:][::-1])
        except Exception as e:
            logging.error(f"Error loading ONNX recognizer model: {e}")
            return

        known_embeddings = []
        known_names = []

        if os.path.exists(EMBEDDINGS_FILE):
            try:
                with open(EMBEDDINGS_FILE, "rb") as f:
                    saved_data = pickle.load(f)
                    existing_names = set(saved_data.get("names", []))
                logging.info(f"Loaded existing embeddings, skipping users: {existing_names}")
            except Exception as e:
                logging.warning(f"Could not load existing embeddings: {e}")
                existing_names = set()
        else:
            existing_names = set()

        logging.info("Starting dataset processing for embedding generation...")
        for person_name in os.listdir(DATASET_PATH):
            person_dir = os.path.join(DATASET_PATH, person_name)
            if not os.path.isdir(person_dir):
                continue

            if person_name in existing_names:
                logging.info(f"Skipping {person_name}, already in saved embeddings.")
                continue

            image_count = 0
            for image_name in os.listdir(person_dir):
                image_path = os.path.join(person_dir, image_name)
                image = cv2.imread(image_path)
                if image is None:
                    logging.warning(f"Could not read image {image_path}")
                    continue

                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                recognizer_input = preprocess_for_recognizer(rgb_image, input_shape)
                embedding = recognizer_session.run([recognizer_output_name], {recognizer_input_name: recognizer_input})[0].flatten()
                known_embeddings.append(embedding)
                known_names.append(person_name)
                image_count += 1
            logging.info(f"Processed {image_count} images for {person_name}.")

        if not known_embeddings:
            logging.info("No new embeddings to add.")
            return

        if existing_names:
            try:
                with open(EMBEDDINGS_FILE, "rb") as f:
                    saved_data = pickle.load(f)
                    known_embeddings = saved_data.get("embeddings", []) + known_embeddings
                    known_names = saved_data.get("names", []) + known_names
            except Exception as e:
                logging.warning(f"Could not merge with existing embeddings: {e}")

        with open(EMBEDDINGS_FILE, "wb") as f:
            pickle.dump({"embeddings": known_embeddings, "names": known_names}, f)

        logging.info(f"Embeddings saved to {EMBEDDINGS_FILE}. Total embeddings: {len(known_embeddings)}")
        reload_embeddings()

if __name__ == '__main__':
    retrain_and_save_embeddings()
