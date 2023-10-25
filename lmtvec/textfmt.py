import io

import numpy as np

from . import dbschema


def convert_from_text(dbfile, fp, chunk=25000):
    db = dbschema.init_db(dbfile)
    cur = db.cursor()
    cur.execute("INSERT INTO metadata VALUES ('chunk_size',?)", (chunk,))
    total_num = 0
    chunk_id = 0
    dim = None
    chunk_words = []
    chunk_vectors = []
    text_wrapper = io.TextIOWrapper(fp, encoding='utf-8')
    for lineno, ln in enumerate(text_wrapper):
        values = ln.rstrip().split(' ', 1)
        if lineno == 0 and len(values) == 2 and values[0].isdigit() and values[1].isdigit():
            continue
        elif dim is None:
            dim = len(values) - 1
            cur.execute("INSERT INTO metadata VALUES ('dim',?)", (dim,))
        if len(chunk_words) >= chunk:
            for i, word in enumerate(chunk_words):
                cur.execute(
                    "INSERT INTO words (id, word, chunk_id, chunk_num) VALUES (?,?,?,?)",
                    (total_num, word, chunk_id, i)
                )
                total_num += 1
            data = dbschema.dump_chunk_content(np.loadtxt(chunk_vectors, dtype='float32'))
            cur.execute("INSERT INTO word_vectors VALUES (?,?)", (chunk_id, data))
            chunk_words.clear()
            chunk_vectors.clear()
            chunk_id += 1
        chunk_words.append(values[0])
        chunk_vectors.append(values[1])
    if len(chunk_words) > 0:
        for i, word in enumerate(chunk_words):
            cur.execute(
                "INSERT INTO words (id, word, chunk_id, chunk_num) VALUES (?,?,?,?)",
                (total_num, word, chunk_id, i)
            )
            total_num += 1
        data = dbschema.dump_chunk_content(np.loadtxt(chunk_vectors, dtype='float32'))
        cur.execute("INSERT INTO word_vectors VALUES (?,?)", (chunk_id, data))
        chunk_words.clear()
        chunk_vectors.clear()
        chunk_id += 1
    cur.execute("INSERT INTO metadata VALUES ('nwords',?)", (total_num,))
    db.commit()
