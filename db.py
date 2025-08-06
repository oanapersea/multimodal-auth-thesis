import sqlite3
from pathlib import Path
import numpy as np
from datetime import datetime

DB_PATH = Path(__file__).parent / "auth.db"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL
        )
    """)

    cur.execute("""
            CREATE TABLE IF NOT EXISTS audio_embeddings (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id       INTEGER NOT NULL,
                orig_id       TEXT    NOT NULL,
                is_augmented  INTEGER NOT NULL DEFAULT 0,
                embedding     BLOB    NOT NULL,
                FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
            )
        """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS face_embeddings (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id       INTEGER NOT NULL,
            orig_id       TEXT    NOT NULL,
            is_augmented  INTEGER NOT NULL DEFAULT 0,
            embedding     BLOB    NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
        )
    """)

    cur.execute("""
           CREATE TABLE IF NOT EXISTS logs (
               id        INTEGER PRIMARY KEY AUTOINCREMENT,
               username  TEXT    NOT NULL,
               method    TEXT    NOT NULL,
               status    TEXT    NOT NULL,      
               timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
           )
       """)

    conn.commit()
    conn.close()


def add_user(username: str):
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()
    cur.execute("INSERT OR IGNORE INTO users(username) VALUES(?)", (username,))
    conn.commit()
    conn.close()


def _get_user_id(username: str) -> int:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT id FROM users WHERE username = ?", (username,))
    row = cur.fetchone()
    if row:
        uid = row[0]
    else:
        cur.execute("INSERT INTO users(username) VALUES(?)", (username,))
        uid = cur.lastrowid
        conn.commit()
    conn.close()
    return uid


def user_exists(username: str) -> bool:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM users WHERE username = ?", (username,))
    exists = cur.fetchone() is not None
    conn.close()
    return exists


def add_audio_embedding(username: str,
                        blob: bytes,
                        orig_id: str,
                        is_augmented: int = 0):
    uid = _get_user_id(username)
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()
    cur.execute("""
        INSERT INTO audio_embeddings
            (user_id, orig_id, is_augmented, embedding)
        VALUES (?,      ?,       ?,            ?)
    """, (uid, orig_id, is_augmented, blob))
    conn.commit()
    conn.close()


def add_face_embedding(username, emb_blob, *, orig_id, is_augmented):
    with sqlite3.connect(DB_PATH) as conn:
        uid = conn.execute(
            "SELECT id FROM users WHERE username=?",
            (username,)
        ).fetchone()[0]
        conn.execute("""
            INSERT INTO face_embeddings
                   (user_id, orig_id, is_augmented, embedding)
            VALUES (?, ?, ?, ?)
        """, (uid, orig_id, is_augmented, emb_blob))
        conn.commit()

def delete_user_data(username: str):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("PRAGMA foreign_keys = ON")  # also delete related data
        cur = conn.cursor()

        row = cur.execute(
            "SELECT id FROM users WHERE username = ?",
            (username,)
        ).fetchone()
        if not row:
            return
        uid = row[0]

        cur.execute("DELETE FROM audio_embeddings WHERE user_id = ?", (uid,))
        cur.execute("DELETE FROM face_embeddings  WHERE user_id = ?", (uid,))
        cur.execute("DELETE FROM users WHERE id = ?", (uid,))

        conn.commit()


def log_attempt(username: str, method: str, ok: bool):
    status = "granted" if ok else "denied"
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute(
        "INSERT INTO logs(username, method, status, timestamp) VALUES (?, ?, ?,?)",
        (username, method, status, ts)
    )
    con.commit()
    con.close()


def get_audio_embeddings(username: str, emb_dim: int = 256) -> list[np.ndarray]:
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()
    cur.execute("""
        SELECT a.embedding 
          FROM audio_embeddings a
          JOIN users u ON u.id = a.user_id
         WHERE u.username = ?
    """, (username,))
    out = []
    for (blob,) in cur.fetchall():
        if isinstance(blob, memoryview):  # sometimes blobs are returned as memory view
            blob = blob.tobytes()
        if len(blob) != emb_dim * 4:
            continue
        vec = np.frombuffer(blob, dtype=np.float32)
        out.append(vec)
    conn.close()
    return out


def get_all_face_rows():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        """
        SELECT f.id,
               f.orig_id,
               f.is_augmented,
               u.username,
               f.embedding
          FROM face_embeddings f
          JOIN users u ON u.id = f.user_id
        """
    ).fetchall()
    conn.close()
    return rows


def get_all_usernames() -> list[str]:
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()
    cur.execute("SELECT username FROM users")
    rows = cur.fetchall()
    conn.close()
    return [r[0] for r in rows]
