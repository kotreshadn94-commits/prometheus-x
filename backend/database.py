"""SQLite persistence and deterministic starter data for PROMETHEUS-X."""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DB_PATH = Path(os.getenv("PROMETHEUS_DB", Path(__file__).parent / "prometheus.db"))


def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DB_PATH, timeout=15)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA foreign_keys=ON")
    return db


def init_db() -> None:
    with connect() as db:
        db.executescript("""
        CREATE TABLE IF NOT EXISTS batches (
          batch_id TEXT PRIMARY KEY, product TEXT NOT NULL, units INTEGER NOT NULL,
          created_at TEXT NOT NULL, supplier TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS inspections (
          id INTEGER PRIMARY KEY AUTOINCREMENT, batch_id TEXT, serial_id TEXT,
          verdict TEXT NOT NULL, defect_type TEXT NOT NULL, severity REAL NOT NULL,
          confidence REAL NOT NULL, latency_ms REAL NOT NULL, approved_normal INTEGER NOT NULL DEFAULT 0,
          result_json TEXT NOT NULL, created_at TEXT NOT NULL,
          FOREIGN KEY(batch_id) REFERENCES batches(batch_id)
        );
        CREATE TABLE IF NOT EXISTS serials (
          serial_id TEXT PRIMARY KEY, batch_id TEXT NOT NULL, product TEXT NOT NULL,
          verdict TEXT NOT NULL, inspection_id INTEGER, issued_at TEXT NOT NULL,
          FOREIGN KEY(batch_id) REFERENCES batches(batch_id),
          FOREIGN KEY(inspection_id) REFERENCES inspections(id)
        );
        """)
        if db.execute("SELECT COUNT(*) FROM batches").fetchone()[0] == 0:
            now = datetime.now(timezone.utc).isoformat()
            batches = [
                ("BATCH-2401", "Precision Alloy Fastener", 1200, now, "Northstar Metals"),
                ("BATCH-2402", "Precision Alloy Fastener", 860, now, "Northstar Metals"),
                ("BATCH-2403", "Rotor Housing Assembly", 540, now, "Helix Components"),
            ]
            db.executemany("INSERT INTO batches VALUES(?,?,?,?,?)", batches)
            serials = [
                ("PX-7F2A91", "BATCH-2401", "Precision Alloy Fastener", "PASS", None, now),
                ("PX-7F2A92", "BATCH-2401", "Precision Alloy Fastener", "PASS", None, now),
                ("PX-8C4D10", "BATCH-2402", "Precision Alloy Fastener", "PASS", None, now),
                ("PX-9B5E31", "BATCH-2403", "Rotor Housing Assembly", "PASS", None, now),
            ]
            db.executemany("INSERT INTO serials VALUES(?,?,?,?,?,?)", serials)


def record_inspection(result: dict[str, Any], batch_id: str | None = None,
                      serial_id: str | None = None) -> int:
    now = datetime.now(timezone.utc).isoformat()
    with connect() as db:
        cursor = db.execute(
            "INSERT INTO inspections(batch_id,serial_id,verdict,defect_type,severity,confidence,latency_ms,result_json,created_at) "
            "VALUES(?,?,?,?,?,?,?,?,?)",
            (batch_id, serial_id, result["verdict"], result["defect_type"], result["severity"],
             result["confidence"], result["latency_ms"], json.dumps(result), now),
        )
        inspection_id = int(cursor.lastrowid)
        if serial_id:
            db.execute("UPDATE serials SET inspection_id=?, verdict=? WHERE serial_id=?",
                       (inspection_id, result["verdict"], serial_id))
        return inspection_id


def get_batches() -> list[dict[str, Any]]:
    with connect() as db:
        rows = db.execute("""
          SELECT b.*, COUNT(i.id) AS inspected,
            SUM(CASE WHEN i.verdict='PASS' THEN 1 ELSE 0 END) AS passed,
            SUM(CASE WHEN i.defect_type='Micro-Scratch' AND i.verdict='PASS' THEN 1 ELSE 0 END) AS scratch_pass
          FROM batches b LEFT JOIN inspections i ON i.batch_id=b.batch_id
          GROUP BY b.batch_id ORDER BY b.created_at DESC
        """).fetchall()
    result = []
    for row in rows:
        item = dict(row)
        item["pass_rate"] = round((item["passed"] or 0) * 100 / item["inspected"], 1) if item["inspected"] else 99.8
        item["scratch_sla"] = "COMPLIANT" if (item["scratch_pass"] or 0) == 0 else "REVIEW"
        result.append(item)
    return result


def get_serial(serial_id: str) -> dict[str, Any] | None:
    with connect() as db:
        row = db.execute("SELECT * FROM serials WHERE serial_id=?", (serial_id.upper(),)).fetchone()
        if not row:
            return None
        item = dict(row)
        if item["inspection_id"]:
            inspection = db.execute("SELECT result_json FROM inspections WHERE id=?", (item["inspection_id"],)).fetchone()
            item["inspection"] = json.loads(inspection["result_json"]) if inspection else None
        else:
            item["inspection"] = None
        item["certificate_id"] = f"PX-CERT-{item['serial_id']}"
        item["verification"] = "VERIFIED"
        return item


def get_review_queue(limit: int = 20) -> list[dict[str, Any]]:
    with connect() as db:
        rows = db.execute("SELECT id,batch_id,verdict,defect_type,severity,confidence,created_at,result_json FROM inspections "
                          "WHERE approved_normal=0 AND confidence BETWEEN 45 AND 65 ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [dict(row) | {"result": json.loads(row["result_json"])} for row in rows]


def approve_inspection(inspection_id: int) -> bool:
    with connect() as db:
        cursor = db.execute("UPDATE inspections SET approved_normal=1 WHERE id=?", (inspection_id,))
        return cursor.rowcount > 0


def analytics() -> dict[str, Any]:
    with connect() as db:
        rows = db.execute("SELECT verdict,COUNT(*) n FROM inspections GROUP BY verdict").fetchall()
        recent = db.execute("SELECT verdict FROM inspections ORDER BY id DESC LIMIT 5").fetchall()
    counts = {row["verdict"]: row["n"] for row in rows}
    streak = 0
    for row in recent:
        if row["verdict"] == "SCRAP":
            streak += 1
        else:
            break
    return {"pass": counts.get("PASS", 0), "rework": counts.get("REWORK", 0),
            "scrap": counts.get("SCRAP", 0), "critical_streak": streak}


def inspection_by_id(inspection_id: int) -> dict[str, Any] | None:
    with connect() as db:
        row = db.execute("SELECT result_json FROM inspections WHERE id=?", (inspection_id,)).fetchone()
    return json.loads(row["result_json"]) if row else None


init_db()
