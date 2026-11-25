"""SQLite persistence and deterministic starter records for the demo installation."""
from pathlib import Path
import sqlite3
from datetime import datetime, timezone

DB_PATH = Path(__file__).resolve().parent / "prometheus_x.db"


def connect():
    db = sqlite3.connect(DB_PATH, timeout=15)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys=ON")
    return db


def init_db():
    with connect() as db:
        db.executescript("""
        CREATE TABLE IF NOT EXISTS batches(
            batch_id TEXT PRIMARY KEY, product TEXT NOT NULL, units_inspected INTEGER NOT NULL,
            passed INTEGER NOT NULL, rework INTEGER NOT NULL, scrap INTEGER NOT NULL,
            micro_scratch_sla REAL NOT NULL, created_at TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS inspections(
            id INTEGER PRIMARY KEY AUTOINCREMENT, batch_id TEXT, serial_id TEXT,
            verdict TEXT NOT NULL, defect_type TEXT NOT NULL, severity REAL NOT NULL,
            confidence REAL NOT NULL, latency_ms REAL NOT NULL, created_at TEXT NOT NULL,
            image_data TEXT, FOREIGN KEY(batch_id) REFERENCES batches(batch_id));
        CREATE TABLE IF NOT EXISTS products(
            serial_id TEXT PRIMARY KEY, batch_id TEXT NOT NULL, product TEXT NOT NULL,
            verdict TEXT NOT NULL, issued_at TEXT NOT NULL, inspection_id INTEGER,
            FOREIGN KEY(batch_id) REFERENCES batches(batch_id));
        CREATE TABLE IF NOT EXISTS approvals(
            id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL,
            features TEXT NOT NULL, note TEXT NOT NULL);
        """)
        now = datetime.now(timezone.utc).isoformat()
        rows = [
            ("PX-2026-041", "Precision Metal Nut", 1280, 1252, 22, 6, 99.2, now),
            ("PX-2026-042", "Precision Metal Nut", 960, 944, 12, 4, 99.5, now),
            ("PX-2026-043", "Alloy Housing", 740, 711, 21, 8, 98.7, now),
        ]
        db.executemany("INSERT OR IGNORE INTO batches VALUES(?,?,?,?,?,?,?,?)", rows)
        serials = [("PX-NUT-000184", "PX-2026-041", "Precision Metal Nut"),
                   ("PX-NUT-000185", "PX-2026-041", "Precision Metal Nut"),
                   ("PX-HSG-000031", "PX-2026-043", "Alloy Housing")]
        for serial, batch, product in serials:
            db.execute("INSERT OR IGNORE INTO products(serial_id,batch_id,product,verdict,issued_at) VALUES(?,?,?,?,?)",
                       (serial, batch, product, "PASS", now))


def record_inspection(data):
    with connect() as db:
        cur = db.execute("""INSERT INTO inspections(batch_id,serial_id,verdict,defect_type,severity,confidence,latency_ms,created_at,image_data)
                            VALUES(?,?,?,?,?,?,?,?,?)""",
                         (data.get("batch_id"), data.get("serial_id"), data["verdict"], data["defect_type"],
                          data["severity"], data["confidence"], data["latency_ms"],
                          datetime.now(timezone.utc).isoformat(), data.get("image_data")))
        return cur.lastrowid


def get_batches():
    with connect() as db:
        return [dict(row) for row in db.execute("SELECT * FROM batches ORDER BY batch_id DESC")]


def get_products():
    with connect() as db:
        return [dict(row) for row in db.execute("SELECT serial_id FROM products ORDER BY serial_id")]


def certificate(serial_id):
    with connect() as db:
        row = db.execute("""SELECT p.*, b.units_inspected, b.micro_scratch_sla, i.defect_type,
                            i.severity, i.confidence, i.created_at AS inspected_at, i.image_data
                            FROM products p JOIN batches b USING(batch_id)
                            LEFT JOIN inspections i ON i.id=p.inspection_id
                            WHERE p.serial_id=?""", (serial_id,)).fetchone()
        return dict(row) if row else None


def log_approval(features, note):
    import json
    with connect() as db:
        db.execute("INSERT INTO approvals(created_at,features,note) VALUES(?,?,?)",
                   (datetime.now(timezone.utc).isoformat(), json.dumps(features), note))


def recent_inspections(limit=30):
    with connect() as db:
        return [dict(r) for r in db.execute("SELECT id,verdict,defect_type,severity,confidence,latency_ms,created_at FROM inspections ORDER BY id DESC LIMIT ?", (limit,))]
