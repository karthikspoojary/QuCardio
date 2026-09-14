import os
import sqlite3
import json
from datetime import datetime

# Default to backend/audit.db or environment variable
DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audit.db")
DATABASE_URL = os.getenv("SQLITE_DB_PATH", DEFAULT_DB_PATH)

def init_db():
    conn = sqlite3.connect(DATABASE_URL)
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
    conn.close()

def log_prediction(patient_id: str, image_hash: str, model_used: str, predicted_class: str, confidence: float, probabilities: dict):
    triage_action = "REFER_TO_CARDIOLOGIST" if confidence < 0.75 else None
    
    conn = sqlite3.connect(DATABASE_URL)
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
    log_id = cursor.lastrowid
    conn.close()
    
    return log_id
