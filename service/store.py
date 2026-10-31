"""Local SQLite records; images live outside the database."""
import json
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(os.getenv('WOUNDWATCH_DATA_DIR', Path(__file__).resolve().parents[1] / 'runtime'))

@contextmanager
def connection():
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / 'images').mkdir(exist_ok=True)
    db = sqlite3.connect(ROOT / 'woundwatch.sqlite', timeout=30)
    db.row_factory = sqlite3.Row
    db.execute('PRAGMA foreign_keys=ON')
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('PRAGMA busy_timeout=30000')
    db.executescript('''
      CREATE TABLE IF NOT EXISTS patients(id TEXT PRIMARY KEY, payload TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS scans(id TEXT PRIMARY KEY, patient_id TEXT NOT NULL REFERENCES patients(id), payload TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS events(id TEXT PRIMARY KEY, payload TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS users(id TEXT PRIMARY KEY, email TEXT UNIQUE NOT NULL,
        name TEXT NOT NULL, password_hash TEXT NOT NULL, role TEXT NOT NULL CHECK(role IN ('patient','clinician')),
        patient_id TEXT UNIQUE REFERENCES patients(id), active INTEGER NOT NULL DEFAULT 1);
      CREATE TABLE IF NOT EXISTS assignments(user_id TEXT REFERENCES users(id),patient_id TEXT REFERENCES patients(id),
        PRIMARY KEY(user_id,patient_id));
      CREATE TABLE IF NOT EXISTS sessions(token_hash TEXT PRIMARY KEY,user_id TEXT REFERENCES users(id),
        csrf TEXT NOT NULL,expires REAL NOT NULL);
      CREATE TABLE IF NOT EXISTS auth_attempts(key TEXT PRIMARY KEY, count INTEGER NOT NULL, reset_at REAL NOT NULL);
      CREATE TABLE IF NOT EXISTS wounds(id TEXT PRIMARY KEY,patient_id TEXT REFERENCES patients(id),payload TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS event_reads(user_id TEXT REFERENCES users(id),event_id TEXT REFERENCES events(id),
        PRIMARY KEY(user_id,event_id));
      CREATE TABLE IF NOT EXISTS audit(id INTEGER PRIMARY KEY,actor TEXT,action TEXT NOT NULL,target TEXT,created_at TEXT NOT NULL);
    ''')
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

def records(table):
    with connection() as db:
        return [json.loads(r['payload']) for r in db.execute(f'SELECT payload FROM {table} ORDER BY rowid DESC')]

def get(table, identifier):
    with connection() as db:
        row = db.execute(f'SELECT payload FROM {table} WHERE id=?', (identifier,)).fetchone()
        return json.loads(row['payload']) if row else None

def save(table, record):
    with connection() as db:
        db.execute(f'INSERT INTO {table}(id,payload) VALUES(?,?) ON CONFLICT(id) DO UPDATE SET payload=excluded.payload',
                   (record['id'], json.dumps(record, allow_nan=False)))
