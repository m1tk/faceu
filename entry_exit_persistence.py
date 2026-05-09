import threading
import queue
import os
import cv2
import time
import glob
import random
import time
import sqlite3

DB_PATH = os.path.join(os.path.dirname(__file__), 'entry_exit_log.db')
entry_exit_persistence = None

class EntryExitPersistenceThread(threading.Thread):
    @staticmethod
    def init_global():
        global entry_exit_persistence
        # Create SQLite connection and pass to persistence thread
        entry_exit_conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        entry_exit_persistence = EntryExitPersistenceThread(entry_exit_conn)
        entry_exit_persistence.start()
        return entry_exit_persistence

    def __init__(self, conn, save_dir="log", queue_size=100):
        super().__init__(daemon=True)
        self.save_dir = save_dir
        self.conn = conn
        self.cursor = self.conn.cursor()
        self.q = queue.Queue(maxsize=queue_size)
        os.makedirs(self.save_dir, exist_ok=True)
        self.running = True
        self._setup_db()

    def run(self):
        while self.running:
            try:
                task = self.q.get(timeout=0.5)
            except queue.Empty:
                continue
            if task[0] == 'add_pending_face':
                _, track_id, image, timestamp = task
                self._add_pending_face(track_id, image, timestamp)
            elif task[0] == 'add_raw_face_image':
                _, track_id, image, timestamp = task
                self._add_raw_face_image(track_id, image, timestamp)
            elif task[0] == 'log_entry_exit_event':
                _, track_id, direction, timestamp = task
                self._log_entry_exit_event(track_id, direction, timestamp)
            elif task[0] == 'cleanup_old_pending_faces':
                _, max_age_seconds = task
                self._cleanup_old_pending_faces(max_age_seconds)
            self.q.task_done()

    def stop(self):
        self.running = False
        self.conn.close()

    def _setup_db(self):
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                employee_id TEXT PRIMARY KEY
            )
            """
        )
        self.conn.execute(
            """
            INSERT OR IGNORE INTO users (employee_id) VALUES ('Unknown')
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS entry_exit (
                id INTEGER PRIMARY KEY,
                user TEXT,
                direction TEXT,
                timestamp REAL,
                synced INTEGER DEFAULT 0,
                FOREIGN KEY(user) REFERENCES users(employee_id)
            )
            """
        )
        self.conn.commit()

    def get_next_track_id(self):
        self.cursor.execute("SELECT MAX(id) FROM entry_exit")
        result = self.cursor.fetchone()
        max_id = result[0] if result[0] is not None else 0
        return max_id

    def add_pending_face(self, track_id, image, timestamp):
        self.q.put(('add_pending_face', track_id, image, timestamp))

    def _add_pending_face(self, track_id, image, timestamp):
        ts = time.strftime('%Y%m%d_%H%M%S', time.localtime(timestamp))
        img_path = os.path.join(self.save_dir, f"track_{track_id}_{ts}.jpg")
        cv2.imwrite(img_path, image)
        self.add_entry_exit(track_id, timestamp)
    
    def add_entry_exit(self, track_id, timestamp):
        # Insert or update the DB row for this track_id
        self.cursor.execute(
            "INSERT OR IGNORE INTO entry_exit (id, user, direction, timestamp) VALUES (?, ?, ?, ?)",
            (track_id, None, "pending", timestamp)
        )
        self.conn.commit()

    def add_raw_face_image(self, track_id, image, timestamp):
        self.q.put(('add_raw_face_image', track_id, image, timestamp))

    def _add_raw_face_image(self, track_id, image, timestamp):
        img_path = os.path.join(self.save_dir, f"raw_{track_id}.jpg")
        cv2.imwrite(img_path, image)
        self.add_entry_exit(track_id, timestamp)

    def process_single_pending_track(self, recognition_worker, consecutive_frames=3):
        self.cursor.execute(
            "SELECT id FROM entry_exit WHERE user is null ORDER BY timestamp ASC LIMIT 1"
        )
        row = self.cursor.fetchone()
        if not row:
            return None
        track_id = row[0]
        img_files = sorted(glob.glob(os.path.join(self.save_dir, f"track_{track_id}_*.jpg")), reverse=True)
        random.shuffle(img_files)
        faces_to_process = img_files[:consecutive_frames]
        images = []
        best_name = 'Unknown'
        for img_path in faces_to_process:
            image = cv2.imread(img_path)
            if image is not None:
                images.append(image)
        if images:
            for i, image in enumerate(images):
                job_id = f"pending_{track_id}_{i}"
                recognition_worker.recognize_async(job_id, image)
            similarities = []
            names = []
            for _ in range(len(images)):
                result = recognition_worker.get_result(block=True, timeout=5)
                if result is not None:
                    _, name, best_sim = result
                    similarities.append(best_sim)
                    names.append(name)
            if similarities:
                name_counts = {}
                name_sims = {}
                for name, sim in zip(names, similarities):
                    if name not in name_counts:
                        name_counts[name] = 0
                        name_sims[name] = []
                    name_counts[name] += 1
                    name_sims[name].append(sim)
                best_name = max(name_counts.keys(), key=lambda x: name_counts[x])
            else:
                best_name = 'Unknown'
        else:
            best_name = 'Unknown'
        self.cursor.execute(
            "UPDATE entry_exit SET user = ? WHERE id = ?",
            (best_name, track_id)
        )
        self.conn.commit()

    def log_entry_exit_event(self, track_id, direction, timestamp):
        self.q.put(('log_entry_exit_event', track_id, direction, timestamp))

    def _log_entry_exit_event(self, track_id, direction, timestamp):
        self.cursor.execute(
            "INSERT INTO entry_exit (id, user, direction, timestamp) VALUES (?, null, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET direction = excluded.direction",
            (track_id, direction, timestamp)
        )
        self.conn.commit()

    def cleanup_old_pending_faces(self, lost_id):
        self.q.put(('cleanup_old_pending_faces', lost_id))

    def _cleanup_old_pending_faces(self, lost_id):
        self.cursor.execute("DELETE FROM entry_exit WHERE id = ? and direction = 'pending'", (lost_id,))
        deleted_count = self.cursor.rowcount
        self.conn.commit()
        if deleted_count > 0:
            for img_path in glob.glob(os.path.join(self.save_dir, f"track_{lost_id}_*.jpg")):
                try:
                    os.remove(img_path)
                except OSError:
                    pass