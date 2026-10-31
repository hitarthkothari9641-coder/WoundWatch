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
