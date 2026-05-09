import numpy as np
import multiprocessing as mp
import cv2
import os

from init import get_recognizer, get_embeddings


# Set multiprocessing start method to 'fork' for lower overhead
try:
    mp.set_start_method("fork")
except RuntimeError:
    pass

def _recognition_worker(input_queue, output_queue):
    recognizer_session, recognizer_input_name, recognizer_output_name = get_recognizer()
    known_embeddings, known_names = get_embeddings()
    while True:
        item = input_queue.get()
        if item is None:
            break
        job_id, face_crop = item
        face_crop = np.ascontiguousarray(face_crop)
        # Preprocess
        face_image = cv2.resize(face_crop, (112, 112))
        face_image = face_image.astype(np.float32) * (2.0/255.0) - 1.0
        recognizer_input = np.expand_dims(np.transpose(face_image, (2, 0, 1)), axis=0)
        curr_emb = recognizer_session.run([recognizer_output_name], {recognizer_input_name: recognizer_input})[0].flatten()
        curr_emb_norm = curr_emb / np.linalg.norm(curr_emb)
        similarities = np.dot(known_embeddings, curr_emb_norm)
        best_match_index = np.argmax(similarities)
        best_sim = similarities[best_match_index]
        name = known_names[best_match_index] if best_sim >= 0.4 else "Unknown"
        output_queue.put((job_id, name, best_sim))

class FaceRecognitionWorker:
    def __init__(self, num_workers=None):
        if num_workers is None:
            num_workers = os.cpu_count() or 1
        self.num_workers = num_workers
        self.input_queue = mp.Queue()
        self.output_queue = mp.Queue()
        self.procs = [mp.Process(target=_recognition_worker, args=(self.input_queue, self.output_queue)) for _ in range(self.num_workers)]
        for p in self.procs:
            p.start()
    def recognize_async(self, job_id, face_crop):
        face_crop = np.ascontiguousarray(face_crop)
        self.input_queue.put((job_id, face_crop))
    def get_result(self, block=False, timeout=None):
        try:
            return self.output_queue.get(block=block, timeout=timeout)
        except Exception:
            return None
    def close(self):
        for _ in range(self.num_workers):
            self.input_queue.put(None)
        for p in self.procs:
            p.join()
