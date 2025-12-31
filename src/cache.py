from __future__ import annotations
import sqlite3
from pathlib import Path
from typing import Optional


def init_cache(db_path: str) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS geocode_cache (
                address TEXT PRIMARY KEY,
                lat REAL,
                lon REAL,
                provider TEXT,
                updated_at TEXT
            )
            """
        )
        conn.commit()


def get_cached(db_path: str, address: str) -> Optional[tuple[float, float, str]]:
    init_cache(db_path)
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute(
            "SELECT lat, lon, provider FROM geocode_cache WHERE address = ?",
            (address,),
        )
        row = cur.fetchone()
        if not row:
            return None
        lat, lon, provider = row
        return float(lat), float(lon), str(provider)


def set_cached(db_path: str, address: str, lat: float, lon: float, provider: str) -> None:
    init_cache(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO geocode_cache (address, lat, lon, provider, updated_at)
            VALUES (?, ?, ?, ?, datetime('now'))
            ON CONFLICT(address) DO UPDATE SET
                lat=excluded.lat,
                lon=excluded.lon,
                provider=excluded.provider,
                updated_at=datetime('now')
            """,
            (address, lat, lon, provider),
        )
        conn.commit()
