import os
import sqlite3
import json
from datetime import datetime

# Default to backend/audit.db or environment variable.
# If the backend/ directory isn't writable (e.g. Docker read-only mount),
# fall back to audit.db in the current working directory.
_backend_dir = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DB_PATH = os.path.join(_backend_dir, "audit.db")
DATABASE_URL = os.getenv("SQLITE_DB_PATH", DEFAULT_DB_PATH)

def _ensure_db_writable():
    """Return a writable DB path — falls back to CWD if backend/ is not writable."""
    path = DATABASE_URL
    parent = os.path.dirname(path) or "."
    if not os.access(parent, os.W_OK):
        fallback = os.path.join(os.getcwd(), "audit.db")
        print(f"[db] WARNING: {parent} not writable, using fallback: {fallback}")
        return fallback
    return path

def init_db():
    # BUG FIX: use context manager so the connection is always closed, even on
    # exception, preventing connection leaks.
    with sqlite3.connect(_ensure_db_writable()) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                patient_id TEXT,
                image_hash TEXT,
                model_used TEXT,
                predicted_class TEXT,
                confidence REAL,
                probabilities TEXT,
                triage_action TEXT
            )
        ''')
        # Compatibility view so queries targeting 'predictions' or 'audit_log' both work
        cursor.execute('''
            CREATE VIEW IF NOT EXISTS predictions AS SELECT * FROM audit_log;
        ''')
        conn.commit()

def log_prediction(patient_id: str, image_hash: str, model_used: str, predicted_class: str, confidence: float, probabilities: dict):
    triage_action = "REFER_TO_CARDIOLOGIST" if confidence < 0.75 else None

    # BUG FIX: use context manager so the connection is always closed, even when
    # an exception is raised mid-insert, preventing connection leaks.
    with sqlite3.connect(_ensure_db_writable()) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO audit_log (
                timestamp, patient_id, image_hash, model_used, predicted_class, confidence, probabilities, triage_action
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            patient_id,
            image_hash,
            model_used,
            predicted_class,
            confidence,
            json.dumps(probabilities),
            triage_action
        ))
        conn.commit()
        return cursor.lastrowid
