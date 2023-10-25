import io
import json
import struct
import logging

import numpy as np

from . import vecdb
from . import dbschema


logger = logging.getLogger(__name__)


class FasttextConst:
    VERSION = 12
    FILEFORMAT_MAGIC = 793712314


MODEL_NAME = {'cbow': 1, 'sg': 2, 'sup': 3}
LOSS_NAME = {'hs': 1, 'ns': 2, 'softmax': 3, 'ova': 4}


def load_header_args(fp):
    magics = struct.unpack('<II', fp.read(8))
    if magics[0] != FasttextConst.FILEFORMAT_MAGIC:
        raise ValueError('Bad magic value')
    version = magics[1]
    if version > FasttextConst.VERSION:
        raise ValueError('Unrecognized version: %s' % version)
    b_args = fp.read(12 * 4 + 8)
    v_args = struct.unpack('<iiiiiiiiiiiid', b_args)
    args = {
        'version': version,
        'dim': v_args[0],
        'ws': v_args[1],
        'epoch': v_args[2],
        'minCount': v_args[3],
        'neg': v_args[4],
        'wordNgrams': v_args[5],
        'loss': v_args[6],
        'model': v_args[7],
        'bucket': v_args[8],
        'minn': v_args[9],
        'maxn': v_args[10],
        'lrUpdateRate': v_args[11],
        't_1e9': int(v_args[12]*1e9),
    }
    if version == 11 and args['model'] == MODEL_NAME['sup']:
        # backward compatibility: old supervised models do not use char ngrams.
        args['maxn'] = 0
    return args


def read0(buf: bytes, fp, size=64):
    idx = -1
    while idx == -1:
        idx = buf.find(b'\0')
        if idx != -1:
            return memoryview(buf)[:idx], buf[idx+1:]
        buf += fp.read(size)


def read_size(buf: bytes, fp, size):
    if len(buf) >= size:
        return memoryview(buf)[:size], buf[size:]
    if not buf:
        return fp.read(size), b''
    buf += fp.read(size - len(buf))
    return buf, b''


def load_dictionary(fp):
    b_header = fp.read(3*4 + 2*8)
    size, nwords, nlabels, ntokens, pruneidx_size = struct.unpack(
        '<iiiqq', b_header)
    buf = b''
    words = []
    for i in range(size):
        b_word, buf = read0(buf, fp)
        word = str(b_word, 'utf-8')
        b_count, buf = read_size(buf, fp, 9)
        count, wtype = struct.unpack('<qb', b_count)
        words.append((word, count, wtype))
    pruneidx = []
    for i in range(pruneidx_size):
        b_count, buf = read_size(buf, fp, 8)
        pruneidx.append(struct.unpack('<ii', b_count))
    return {
        '_buf': buf,
        'nwords': nwords,
        'nlabels': nlabels,
        'ntokens': ntokens,
        'words': words,
        'pruneidx': pruneidx,
    }


def load_dense_matrix(fp, db, buf, chunk, table_name='word_vectors'):
    cur = db.cursor()
    b_header, buf = read_size(buf, fp, 8*2)
    m, n = struct.unpack('<qq', b_header)
    logger.info('Matrix: %s, %s' % (m, n))
    chunk_id = 0
    chunk_num = 0
    chunk_vectors = io.BytesIO()
    for i in range(m):
        if chunk_num >= chunk:
            arr = np.frombuffer(chunk_vectors.getbuffer(), dtype='<f4').reshape(
                (chunk_num, n), order='C')
            data = dbschema.dump_chunk_content(arr)
            cur.execute("INSERT INTO {} VALUES (?,?)".format(table_name), (chunk_id, data))
            chunk_num = 0
            chunk_vectors = io.BytesIO()
            chunk_id += 1
        b_row, buf = read_size(buf, fp, 4*n)
        chunk_vectors.write(b_row)
        chunk_num += 1
    if chunk_num > 0:
        arr = np.frombuffer(chunk_vectors.getbuffer(), dtype='<f4').reshape(
            (chunk_num, n), order='C')
        data = dbschema.dump_chunk_content(arr)
        cur.execute("INSERT INTO {} VALUES (?,?)".format(table_name), (chunk_id, data))
        chunk_id += 1
    return buf


def load_pq_data(fp, buf):
    ksub = 1 << 8
    b_header, buf = read_size(buf, fp, 4*4)
    header_params = struct.unpack('<iiii', b_header)
    params = {
        'dim': header_params[0],
        'nsubq': header_params[1],
        'dsub': header_params[2],
        'lastdsub': header_params[3],
    }
    size = params['dim'] * ksub
    logger.info("ProductQuantizer (%s): %s", size, params)
    b_centroids, buf = read_size(buf, fp, 4*size)
    centroids = np.frombuffer(b_centroids, dtype='<f4')
    return params, centroids


def load_quant_matrix(fp, db, buf, table_name='word_vectors'):
    cur = db.cursor()
    b_header, buf = read_size(buf, fp, 1+8+8+4)
    qnorm, m, n, codesize = struct.unpack('<bqqi', b_header)
    cur.execute(
        "INSERT INTO {} VALUES (?,?)".format(table_name),
        ('params', json.dumps({
            'qnorm': qnorm,
            'm': m,
            'n': n,
            'codesize': codesize,
        }).encode('utf-8'))
    )
    b_codes, buf = read_size(buf, fp, codesize)
    codes = np.frombuffer(b_codes, dtype='B')
    cur.execute(
        "INSERT INTO {} VALUES (?,?)".format(table_name),
        ('codes', dbschema.dump_chunk_content(codes))
    )
    pq_params, pq_centroids = load_pq_data(fp, buf)
    cur.execute(
        "INSERT INTO {} VALUES (?,?)".format(table_name),
        ('pq_params', json.dumps(pq_params).encode('utf-8'))
    )
    cur.execute(
        "INSERT INTO {} VALUES (?,?)".format(table_name),
        ('pq_centroids', dbschema.dump_chunk_content(pq_centroids))
    )
    if qnorm:
        b_norm_codes, buf = read_size(buf, fp, m)
        norm_codes = np.frombuffer(b_codes, dtype='B')
        npq_params, npq_centroids = load_pq_data(fp, buf)
        cur.execute(
            "INSERT INTO {} VALUES (?,?)".format(table_name),
            ('norm_codes', dbschema.dump_chunk_content(norm_codes))
        )
        cur.execute(
            "INSERT INTO {} VALUES (?,?)".format(table_name),
            ('npq_params', json.dumps(npq_params).encode('utf-8'))
        )
        cur.execute(
            "INSERT INTO {} VALUES (?,?)".format(table_name),
            ('npq_centroids', dbschema.dump_chunk_content(npq_centroids))
        )
    return buf


def compute_word_vectors(dbfile, db, chunk):
    model = vecdb.TextVectorModel(dbfile, cache_size=128)
    cur = db.cursor()
    cur.execute("SELECT word FROM words ORDER BY id")
    words = [row[0] for row in cur]
    chunk_id = 0
    chunk_vectors = []
    for i, word in enumerate(words):
        if len(chunk_vectors) >= chunk:
            logger.info(" %d/%d", i, len(words))
            arr = np.asarray(chunk_vectors, dtype='f4')
            data = dbschema.dump_chunk_content(arr)
            cur.execute("INSERT INTO word_vectors VALUES (?,?)", (chunk_id, data))
            db.commit()
            chunk_vectors.clear()
            chunk_id += 1
        chunk_vectors.append(model.get_subword_vector(word))
    if len(chunk_vectors) > 0:
        arr = np.asarray(chunk_vectors, dtype='f4')
        data = dbschema.dump_chunk_content(arr)
        cur.execute("INSERT INTO word_vectors VALUES (?,?)", (chunk_id, data))
        db.commit()
        chunk_id += 1
    logger.info("Done.")


def convert_from_fasttext_binary(dbfile, fp, chunk=25000):
    logger.info("Init db...")
    db = dbschema.init_db(dbfile)
    cur = db.cursor()
    cur.execute("DELETE FROM metadata")
    cur.execute("DELETE FROM words")
    cur.execute("DELETE FROM pruneidx")
    cur.execute("DELETE FROM word_vectors")
    cur.execute("DELETE FROM input_matrix")
    cur.execute("DELETE FROM output_matrix")
    cur.execute("DELETE FROM input_quant")
    cur.execute("DELETE FROM output_quant")
    cur.execute("INSERT INTO metadata VALUES ('chunk_size',?)", (chunk,))
    logger.info("Loading header...")
    header_args = load_header_args(fp)
    for key, value in header_args.items():
        cur.execute("INSERT INTO metadata VALUES (?,?)", (key, value))
    logger.info("Args: %s", header_args)
    logger.info("Loading dictionary...")
    dict_args = load_dictionary(fp)
    buf = dict_args.pop('_buf')
    for key in ('nwords', 'nlabels', 'ntokens'):
        cur.execute("INSERT INTO metadata VALUES (?,?)", (key, dict_args[key]))
    logger.info(
        "nwords: %s, nlabels: %s, ntokens: %s",
        dict_args['nwords'], dict_args['nlabels'], dict_args['ntokens']
    )
    total_num = 0
    chunk_id = 0
    chunk_words = []
    for row in dict_args['words']:
        if len(chunk_words) >= chunk:
            for i, (word, count, wtype) in enumerate(chunk_words):
                cur.execute("INSERT INTO words VALUES (?,?,?,?,?,?)", (
                    total_num, word, count, wtype, chunk_id, i
                ))
                total_num += 1
            chunk_words.clear()
            chunk_id += 1
        chunk_words.append(row)
    if len(chunk_words) > 0:
        for i, (word, count, wtype) in enumerate(chunk_words):
            cur.execute("INSERT INTO words VALUES (?,?,?,?,?,?)", (
                total_num, word, count, wtype, chunk_id, i
            ))
            total_num += 1
    for row in dict_args['pruneidx']:
        cur.execute("INSERT INTO pruneidx VALUES (?,?)", row)
    b_quant_input, buf = read_size(buf, fp, 1)
    quant_input = b_quant_input[0]
    logger.info("quant_input=%s", quant_input)
    cur.execute("INSERT INTO metadata VALUES ('quant_input',?)", (quant_input,))
    db.commit()
    if quant_input == 0:
        logger.info("Loading input dense matrix...")
        buf = load_dense_matrix(fp, db, buf, chunk, 'input_matrix')
    else:
        logger.info("Loading input quant matrix...")
        buf = load_quant_matrix(fp, db, buf, 'input_quant')
    db.commit()
    b_qout, buf = read_size(buf, fp, 1)
    quant_output = b_qout[0]
    logger.info("quant_output=%s", quant_output)
    cur.execute("INSERT INTO metadata VALUES ('quant_output',?)", (quant_output,))
    if quant_output == 0:
        logger.info("Loading output dense matrix...")
        load_dense_matrix(fp, db, buf, chunk, 'output_matrix')
    else:
        logger.info("Loading output quant matrix...")
        load_quant_matrix(fp, db, buf, 'output_quant')
    db.commit()
    logger.info("Pre-computing word vectors...")
    compute_word_vectors(dbfile, db, chunk)
