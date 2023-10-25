import io
import sqlite3

import numpy as np


def init_db(dbfile):
    db = sqlite3.connect(dbfile)
    cur = db.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS metadata ("
                "key TEXT PRIMARY KEY,"
                "value INTEGER"
                ")")
    cur.execute("CREATE TABLE IF NOT EXISTS words ("
                "id INTEGER PRIMARY KEY,"
                "word TEXT UNIQUE,"
                "freq INTEGER,"
                "type INTEGER NOT NULL DEFAULT 0,"
                "chunk_id INTEGER,"
                "chunk_num INTEGER"
                ")")
    cur.execute("CREATE TABLE IF NOT EXISTS pruneidx ("
                "key INTEGER PRIMARY KEY,"
                "value INTEGER"
                ")")
    cur.execute("CREATE TABLE IF NOT EXISTS word_vectors ("
                "id INTEGER PRIMARY KEY,"
                "chunk_data BLOB"
                ")")
    cur.execute("CREATE TABLE IF NOT EXISTS input_matrix ("
                "id INTEGER PRIMARY KEY,"
                "chunk_data BLOB"
                ")")
    cur.execute("CREATE TABLE IF NOT EXISTS output_matrix ("
                "id INTEGER PRIMARY KEY,"
                "chunk_data BLOB"
                ")")
    cur.execute("CREATE TABLE IF NOT EXISTS input_quant ("
                "key TEXT PRIMARY KEY,"
                "data BLOB"
                ")")
    cur.execute("CREATE TABLE IF NOT EXISTS output_quant ("
                "key TEXT PRIMARY KEY,"
                "data BLOB"
                ")")
    db.commit()
    return db


def dump_chunk_content(arr):
    npy = io.BytesIO()
    #np.savez_compressed(npy, d=arr)
    np.save(npy, arr)
    return npy.getvalue()

