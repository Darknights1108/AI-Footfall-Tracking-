"""
CLI: drop and recreate the footfall database tables.

Example
-------
    python scripts/reset_db.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config, database


def main() -> int:
    database.reset_db()
    print(f"Database reset complete: {config.DB_PATH}")
    print("Tables recreated: footfall_events, processing_runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
