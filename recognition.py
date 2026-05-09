import cv2
import numpy as np
import time
import face_recognition
import fiaq
import mediapipe as mp
from init import (
    get_recognizer, get_embeddings
)
from tracker import FaceSortTracker

class FaceRecognizer:
    def __init__(self, settings, log_store, recognition_worker):
        self.settings = settings
        self.fps = settings.get('fps', 5)
        self.recognizer_input_shape = tuple(settings.get('recognizer_input_shape', (112, 112)))
        self.tracker = FaceSortTracker(
            max_age=settings.get('tracker_max_age', 5),
            min_hits=settings.get('tracker_min_hits', 1),
            iou_threshold=settings.get('tracker_iou_threshold', 0.15)
        )
        self.track_id_to_face = {}
        self.prev_frame_gray = None
        self.current_state = 1  # 1: DETECTING_MOVEMENT, 2: DETECTING_FACES
        self.last_face_detected_time = None
        self.last_motion_check_time = 0
        # For entry/exit detection
        self.entry_exit_line = settings.get('entry_exit_line', {'y': 200, 'orientation': 'horizontal'})
        self.track_id_last_centers = {}  # track_id: (cx, cy)
        # Persistent entry/exit logger
        self.entry_exit_persistence = log_store
        self.recognition_worker = recognition_worker
        self._pending_recognition = {}  # track_id: job_id
        self._pending_results = {}      # job_id: (track_id, ...)
        
        # Performance optimizations - cache frequently accessed values
        self.motion_detection_interval = settings.get('motion_detection_interval', 0.5)
        self.motion_fps = settings.get('motion_fps', 2)  # Separate FPS for motion detection
        self.movement_threshold = settings.get('movement_threshold', 1500)
        self.no_face_timeout = settings.get('no_face_timeout', 10.0)
        self.consecutive_frames = settings.get('consecutive_frames', 3)
        self.recognition_threshold = settings.get('recognition_threshold', 0.4)
        self.fiaq_threshold = settings.get('fiaq_threshold', 0.4)
        self.min_face_size = settings.get('min_face_size', 50)
        self.detector_conf_thresh = settings.get('detector_conf_thresh', 0.5)
        self.movement_blur_size = tuple(settings.get('movement_blur_size', (7, 7)))
        
        # Pre-compute frame times
        self.frame_time = 1.0 / max(self.fps, 1)
        self.motion_frame_time = 1.0 / max(self.motion_fps, 1)  # Separate frame time for motion detection
        
        # Cache model sessions for performance
        self._recognizer = get_recognizer()
        self._embeddings = get_embeddings()
        
        # Initialize MediaPipe pose detection
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=self.detector_conf_thresh,
            min_tracking_confidence=self.detector_conf_thresh
        )
        
        # FPS tracking optimizations
        self._fps_last_time = 0.0
        self._fps_counter = 0
        self._fps_value = 0.0
        
        # Pre-compute commonly used constants
        self.recognizer_input_w = self.recognizer_input_shape[1]
        self.recognizer_input_h = self.recognizer_input_shape[0]
        
        # Pre-allocate arrays for better memory management
        self._temp_gray = None
        self._temp_resized = None
        
        # Pre-allocate reusable buffers for recognizer preprocessing
        self._recognizer_resize_buf = np.empty((self.recognizer_input_h, self.recognizer_input_w, 3), dtype=np.uint8)
        self._recognizer_float_buf = np.empty((self.recognizer_input_h, self.recognizer_input_w, 3), dtype=np.float32)

        self.detection_mode = settings.get('detection_mode', 'line')
        self.camera_type = settings.get('camera_type', 'entry')
        
    def process_frame(self, frame, now):
        if self._fps_last_time == 0.0:
            self._fps_last_time = now
        self._fps_counter += 1
        if now - self._fps_last_time >= 1.0:
            self._fps_value = self._fps_counter / (now - self._fps_last_time)
            self._fps_last_time = now
            self._fps_counter = 0
            
        if self.current_state == 1:  # DETECTING_MOVEMENT
            self.entry_exit_persistence.process_single_pending_track(self.recognition_worker, self.consecutive_frames)
            if now - self.last_motion_check_time >= self.motion_detection_interval:
                gray, movement_score = self.detect_movement(frame, self.prev_frame_gray)
                self.last_motion_check_time = now
                if self.prev_frame_gray is not None and movement_score > self.movement_threshold:
                    self.current_state = 2
                    self.last_face_detected_time = now
                    self.track_id_to_face.clear()
                self.prev_frame_gray = gray
            # Visual indicator for movement detection mode
            cv2.putText(frame, "Mode: Detecting Movement", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(frame, f"FPS: {self._fps_value:.1f}", (frame.shape[1] - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        elif self.current_state == 2:  # DETECTING_FACES
            self.track_id_to_face, self.last_face_detected_time = self.detect_faces(
                frame, now, self.track_id_to_face, self.last_face_detected_time
            )
            # Check timeout for switching back to movement detection
            if self.last_face_detected_time and now - self.last_face_detected_time > self.no_face_timeout:
                self.current_state = 1
                self.prev_frame_gray = None
            cv2.putText(frame, "Mode: Detecting Faces", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Tracked Faces: {len(self.track_id_to_face)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            # Show timeout countdown
            if self.last_face_detected_time:
                time_since_detection = now - self.last_face_detected_time
                remaining = max(0, self.no_face_timeout - time_since_detection)
                cv2.putText(frame, f"Timeout: {remaining:.1f}s", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"FPS: {self._fps_value:.1f}", (frame.shape[1] - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Calculate time to sleep to respect FPS - use appropriate FPS based on current state
        elapsed = time.time() - now
        if self.current_state == 1:  # DETECTING_MOVEMENT
            time_to_sleep = max(0, self.motion_frame_time - elapsed)
        else:  # DETECTING_FACES
            time_to_sleep = max(0, self.frame_time - elapsed)
        return frame, time_to_sleep

    def detect_movement(self, frame_rgb, prev_frame_gray):
        # Optimized movement detection with pre-cached blur size
        small_frame = cv2.resize(frame_rgb, (320, 240), interpolation=cv2.INTER_LINEAR)
        gray = cv2.cvtColor(small_frame, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, self.movement_blur_size, 0)
        movement_score = 0
        if prev_frame_gray is not None:
            frame_delta = cv2.absdiff(prev_frame_gray, gray)
            thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
            movement_score = np.count_nonzero(thresh)
        return gray, movement_score

    def detect_faces(self, frame_rgb, loop_start_time, track_id_to_face, last_face_detected_time):
        raw_frame = frame_rgb.copy()
        # Convert RGB to BGR for MediaPipe
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        
        # Detect pose
        results = self.pose.process(frame_bgr)
        
        frame_height, frame_width = frame_rgb.shape[:2]
        detections = []
        
        if results.pose_landmarks:
            # Extract face region from pose landmarks
            landmarks = results.pose_landmarks.landmark
            
            # Get key face landmarks from pose (nose, left/right eye, left/right mouth)
            nose = landmarks[self.mp_pose.PoseLandmark.NOSE]
            left_eye = landmarks[self.mp_pose.PoseLandmark.LEFT_EYE]
            right_eye = landmarks[self.mp_pose.PoseLandmark.RIGHT_EYE]
            left_mouth = landmarks[self.mp_pose.PoseLandmark.MOUTH_LEFT]
            right_mouth = landmarks[self.mp_pose.PoseLandmark.MOUTH_RIGHT]
            
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
                
                if face_w >= self.min_face_size and face_h >= self.min_face_size:
                    # Make bounding box more square for better face recognition
                    if face_w > face_h:
                        diff = face_w - face_h
                        y1 = max(0, y1 - diff // 2)
                        y2 = min(frame_height, y2 + diff // 2)
                    elif face_h > face_w:
                        diff = face_h - face_w
                        x1 = max(0, x1 - diff // 2)
                        x2 = min(frame_width, x2 + diff // 2)
                    
                    detections.append([x1, y1, x2, y2])
        
        tracks = self.tracker.update(np.array(detections) if detections else np.empty((0, 5)))
        
        # Remove track IDs that are no longer in vision
        current_track_ids = set(int(trk[4]) for trk in tracks)
        previous_track_ids = set(track_id_to_face.keys())
        lost_track_ids = previous_track_ids - current_track_ids
        for lost_id in lost_track_ids:
            # Optionally, you could pass a custom max_age_seconds for immediate cleanup
            self.entry_exit_persistence.cleanup_old_pending_faces(lost_id)
            # Remove from local tracking dict
            if lost_id in track_id_to_face:
                del track_id_to_face[lost_id]
        
        # Pre-compute entry/exit line configurations
        entry_line = self.settings.get('entry_line')
        exit_line = self.settings.get('exit_line')
        
        # Draw entry and exit lines only in line mode
        if self.detection_mode == 'line':
            if entry_line:
                if entry_line.get('orientation', 'horizontal') == 'horizontal':
                    y = entry_line.get('y', 100)
                    cv2.line(frame_rgb, (0, y), (frame_width, y), (40, 200, 40), 2)
                    cv2.putText(frame_rgb, "ENTRY", (10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 200, 40), 2)
                else:
                    x = entry_line.get('x', 100)
                    cv2.line(frame_rgb, (x, 0), (x, frame_height), (40, 200, 40), 2)
                    cv2.putText(frame_rgb, "ENTRY", (x + 5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 200, 40), 2)
            if exit_line:
                if exit_line.get('orientation', 'horizontal') == 'horizontal':
                    y = exit_line.get('y', 200)
                    cv2.line(frame_rgb, (0, y), (frame_width, y), (200, 40, 120), 2)
                    cv2.putText(frame_rgb, "EXIT", (10, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 40, 120), 2)
                else:
                    x = exit_line.get('x', 200)
                    cv2.line(frame_rgb, (x, 0), (x, frame_height), (200, 40, 120), 2)
                    cv2.putText(frame_rgb, "EXIT", (x + 5, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 40, 120), 2)
                
        # Optimized entry/exit logic for both lines
        for trk in tracks:
            x1, y1, x2, y2, track_id = map(int, trk)
            cx, cy = (x1 + x2) >> 1, (y1 + y2) >> 1  # Bit shift for faster division
            tface = track_id_to_face.get(track_id)
            
            # Initialize tracking data if this is a new face
            if track_id not in track_id_to_face:
                track_id_to_face[track_id] = TrackedFace(track_id, (x1, y1, x2, y2), loop_start_time)
                tface = track_id_to_face[track_id]
            else:
                track_id_to_face[track_id].update((x1, y1, x2, y2), loop_start_time)
                tface = track_id_to_face[track_id]
            
            # Initialize logging attributes only once
            if not hasattr(tface, 'current_zone'):
                tface.current_zone = None  # Track which zone the face is currently in

            # Helper function to determine which zone a face center is in
            def get_current_zone(cx, cy):
                current_zone = None
                
                # Check entry line zone
                if entry_line:
                    orient = entry_line.get('orientation', 'horizontal')
                    direction = entry_line.get('direction', 'top')
                    if orient == 'horizontal':
                        y = entry_line.get('y', 100)
                        if direction == 'top':
                            if cy > y:
                                current_zone = 'before_entry'
                            else:
                                current_zone = 'after_entry'
                        else:  # direction == 'bottom'
                            if cy < y:
                                current_zone = 'before_entry'
                            else:
                                current_zone = 'after_entry'
                    else:  # vertical
                        x = entry_line.get('x', 100)
                        if direction == 'left':
                            if cx < x:
                                current_zone = 'before_entry'
                            else:
                                current_zone = 'after_entry'
                        else:  # direction == 'right'
                            if cx > x:
                                current_zone = 'before_entry'
                            else:
                                current_zone = 'after_entry'
                
                # Check exit line zone (only if no entry line zone determined)
                if current_zone is None and exit_line:
                    orient = exit_line.get('orientation', 'horizontal')
                    direction = exit_line.get('direction', 'bottom')
                    if orient == 'horizontal':
                        y = exit_line.get('y', 200)
                        if direction == 'top':
                            if cy > y:
                                current_zone = 'before_exit'
                            else:
                                current_zone = 'after_exit'
                        else:  # direction == 'bottom'
                            if cy < y:
                                current_zone = 'before_exit'
                            else:
                                current_zone = 'after_exit'
                    else:  # vertical
                        x = exit_line.get('x', 200)
                        if direction == 'left':
                            if cx < x:
                                current_zone = 'before_exit'
                            else:
                                current_zone = 'after_exit'
                        else:  # direction == 'right'
                            if cx > x:
                                current_zone = 'before_exit'
                            else:
                                current_zone = 'after_exit'
                
                return current_zone

            # Get current zone for this face
            if self.detection_mode == 'line':
                new_zone = get_current_zone(cx, cy)
                line_crossed = False
                
                # Check for zone transitions and log events
                if tface.current_zone is not None and new_zone != tface.current_zone:
                    # Entry detection: moved from before_entry to after_entry
                    if tface.current_zone == 'before_entry' and new_zone == 'after_entry' and not line_crossed:
                        self.entry_exit_persistence.log_entry_exit_event(
                            track_id=track_id,
                            direction='entered',
                            timestamp=loop_start_time
                        )
                        line_crossed = True
                    
                    # Exit detection: moved from before_exit to after_exit
                    elif tface.current_zone == 'before_exit' and new_zone == 'after_exit' and not line_crossed:
                        self.entry_exit_persistence.log_entry_exit_event(
                            track_id=track_id,
                            direction='exited',
                            timestamp=loop_start_time
                        )
                        line_crossed = True
                
                # Update the face's current zone
                if new_zone is not None:
                    tface.current_zone = new_zone
                            
                # Visual feedback when line is crossed
                if line_crossed:
                    cv2.circle(frame_rgb, (cx, cy), 10, (0, 255, 255), 3)  # Yellow circle for line crossing
            elif self.detection_mode == 'camera':
                # Camera type mode: log entry/exit based on camera type
                if not tface.entry_exit_logged:
                    direction = 'entered' if self.camera_type == 'entry' else 'exited'
                    self.entry_exit_persistence.log_entry_exit_event(
                        track_id=track_id,
                        direction=direction,
                        timestamp=loop_start_time
                    )
                    tface.entry_exit_logged = True
            
            self.track_id_last_centers[track_id] = (cx, cy)
        # Only update last_face_detected_time if we have NEW detections, not just continuing tracks
        if len(detections) > 0:
            last_face_detected_time = loop_start_time
            
        # Optimized face drawing and recognition loop
        for track_id, tface in track_id_to_face.items():
            x1, y1, x2, y2 = tface.bbox
            box_title = f"ID {track_id}"
            # Save only one raw face image per tracked face
            if not tface.raw_image_saved:
                self.entry_exit_persistence.add_raw_face_image(track_id=track_id, image=raw_frame, timestamp=loop_start_time)
                tface.raw_image_saved = True
            face_crop = frame_rgb[y1:y2, x1:x2]
            if face_crop.size == 0: continue
            landmarks, face_image_aligned = self.preprocess_for_recognizer(face_crop)
            if landmarks is None or fiaq.assess_face_quality(face_image_aligned, landmarks) < self.fiaq_threshold:
                cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (0, 140, 255), 2)
                cv2.putText(frame_rgb, f"{box_title} | Low Quality", (x1 + 5, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2)
                continue
            if tface.raw_image_saved and not tface.optimal_raw_image:
                self.entry_exit_persistence.add_raw_face_image(track_id=track_id, image=raw_frame, timestamp=loop_start_time)
                tface.optimal_raw_image = True
            self.entry_exit_persistence.add_pending_face(
                track_id=track_id,
                image=face_image_aligned,
                timestamp=loop_start_time
            )
            cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (255, 165, 0), 2)
            cv2.putText(frame_rgb, f"{box_title} | Stored for Recognition", (x1 + 5, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2)
                
        return track_id_to_face, last_face_detected_time

    def preprocess_for_recognizer(self, face_image_rgb):
        # Use pre-allocated buffers for better performance
        cv2.resize(face_image_rgb, (self.recognizer_input_w, self.recognizer_input_h), dst=self._recognizer_resize_buf)
        face_image_aligned, landmarks = align_face(self._recognizer_resize_buf)
        np.multiply(face_image_aligned.astype(np.float32), 2.0/255.0, out=self._recognizer_float_buf)
        np.subtract(self._recognizer_float_buf, 1.0, out=self._recognizer_float_buf)
        return (
            landmarks,
            face_image_aligned
        )

class TrackedFace:
    
    def __init__(self, face_id, bbox, timestamp):
        self.face_id = face_id
        self.bbox = bbox
        self.last_update = timestamp
        self.frame_count = 1
        self.similarity = 0.0
        self.entry_exit_logged = False
        self.last_overlay_frame = None
        self.current_zone = None
        self.raw_image_saved = False
        self.optimal_raw_image = False
    
    def update(self, bbox, timestamp):
        self.bbox = bbox
        self.last_update = timestamp
        self.frame_count += 1
    
    def reset(self, bbox, timestamp):
        self.bbox = bbox
        self.last_update = timestamp
        self.frame_count = 1
        self.similarity = 0.0

def align_face(face_image_rgb):
    """Optimized face alignment with early return and vectorized operations"""
    landmarks = face_recognition.face_landmarks(face_image_rgb)
    if not landmarks: 
        return (face_image_rgb, None)
    
    # Vectorized eye center computation
    left_eye = np.mean(landmarks[0]['left_eye'], axis=0)
    right_eye = np.mean(landmarks[0]['right_eye'], axis=0)
    
    # Compute angle and center efficiently
    eye_diff = right_eye - left_eye
    angle = np.degrees(np.arctan2(eye_diff[1], eye_diff[0]))
    eyes_center = (left_eye + right_eye) * 0.5
    
    # Get rotation matrix and apply transformation
    M = cv2.getRotationMatrix2D(tuple(eyes_center), angle, scale=1.0)
    aligned_face = cv2.warpAffine(face_image_rgb, M, 
                                 (face_image_rgb.shape[1], face_image_rgb.shape[0]), 
                                 flags=cv2.INTER_CUBIC)
    
    return (aligned_face, landmarks)
