import os
import pickle
import threading
import numpy as np
import onnxruntime as ort
from dataclasses import dataclass
from typing import Optional, Any, List


@dataclass
class SharedState:
    recognizer_session: Optional[Any] = None
    recognizer_input_name: Optional[str] = None
    recognizer_output_name: Optional[str] = None
    known_embeddings: Optional[np.ndarray] = None
    known_names: Optional[List[str]] = None
    embeddings_file: Optional[str] = None
    model_dir: Optional[str] = None

# Shared lock for embeddings
embeddings_lock = threading.Lock()

# Shared model and embeddings (initialized on first use)
shared_state = SharedState()

def _create_session(model_path: str, providers: Optional[List[str]] = None):
    if providers is None:
        providers = ["CPUExecutionProvider"]
    sess_options = ort.SessionOptions()
    return ort.InferenceSession(model_path, sess_options=sess_options, providers=providers)

def initialize_shared(model_dir, embeddings_file, recognizer_model_name='w600k_r50.onnx'):
    if shared_state.recognizer_session is None:
        recognizer_model = os.path.join(model_dir, recognizer_model_name)
        shared_state.recognizer_session = _create_session(recognizer_model)
        shared_state.recognizer_input_name = shared_state.recognizer_session.get_inputs()[0].name
        shared_state.recognizer_output_name = shared_state.recognizer_session.get_outputs()[0].name
        shared_state.model_dir = model_dir
        shared_state.embeddings_file = embeddings_file
    reload_embeddings()

def reload_embeddings():
    with embeddings_lock:
        embeddings_file = shared_state.embeddings_file
        if not os.path.exists(embeddings_file):
            return
        with open(embeddings_file, 'rb') as f:
            data = pickle.load(f)
        shared_state.known_embeddings = np.array([
            emb / np.linalg.norm(emb) if np.linalg.norm(emb) > 0 else emb for emb in data['embeddings']
        ], dtype=np.float32)
        shared_state.known_names = data['names']

def get_recognizer():
    return shared_state.recognizer_session, shared_state.recognizer_input_name, shared_state.recognizer_output_name

def get_embeddings():
    return shared_state.known_embeddings, shared_state.known_names
