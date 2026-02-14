import sqlite3
from pathlib import Path

SCHEMA_FILE = Path(__file__).with_name("schema_v2.sql")
DB_PATH = "insider_database_v2.db"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")

    with open(SCHEMA_FILE, "r") as f:
        conn.executescript(f.read())

    conn.commit()
    conn.close()
    print("Database initialized")


if __name__ == "__main__":
    init_db()
