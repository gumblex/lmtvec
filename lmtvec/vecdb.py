import io
import json
import heapq
import sqlite3
import cachetools

import numpy as np


class StringConst:
    EOS = "</s>"
    BOW = "<"
    EOW = ">"


def _load_chunk_content(content):
    return np.load(io.BytesIO(content))


def dict_hash(s):
    arr = np.frombuffer(s.encode('utf-8'), dtype='i1').astype('u4')
    h = np.array(2166136261, dtype='u4')
    const = np.array(16777619, dtype='u4')
    for i in range(arr.shape[0]):
        h ^= arr[i]
        h *= const
    return int(h)


class TextVectorModel:
    _matrix_table = ('word_vectors', 'input_matrix', 'output_matrix')

    def __init__(self, dbname, cache_size=32):
        self.db = sqlite3.connect(dbname)
        self.last_chunk_id = None
        self.last_chunk = None
        self.metadata = {}
        self.pruneidx = {}
        self._input_quant = {}
        self.chunk_cache = cachetools.LRUCache(cache_size)
        self._words = {}
        self._words_loaded = False
        cur = self.db.cursor()
        cur.execute("SELECT key, value FROM metadata")
        for key, value in cur:
            self.metadata[key] = value
        cur.execute("SELECT key, value FROM pruneidx")
        for key, value in cur:
            self.pruneidx[key] = value

    def get_dimension(self):
        """Get the dimension (size) of a lookup vector (hidden layer)."""
        return self.metadata['dim']

    def get_dict_word_vector(self, word):
        """Get the vector representation of dictionary word."""
        cur = self.db.cursor()
        cur.execute("SELECT chunk_id, chunk_num FROM words w WHERE word=?", (word,))
        row = cur.fetchone()
        if row is None:
            return None
        return np.copy(self._get_matrix_chunk(1, row[0])[row[1]])

    def split_subword(self, word):
        if word == StringConst.EOS:
            return []
        word = StringConst.BOW + word + StringConst.EOW
        len_word = len(word)
        minn = self.metadata['minn']
        maxn = self.metadata['maxn']
        bucket = self.metadata['bucket']
        subwords = []
        hashes = []
        for pos in range(len(word)):
            for n in range(1, maxn + 1):
                if pos + n - 1 >= len_word:
                    break
                if n >= minn:
                    subword = word[pos:pos + n]
                    h = dict_hash(subword) % bucket
                    subwords.append(subword)
                    hashes.append(h)
        return (subwords, hashes)

    def _get_matrix_chunk(self, matrix_id, chunk_id):
        arr = self.chunk_cache.get((matrix_id, chunk_id))
        if arr is not None:
            return arr
        cur = self.db.cursor()
        cur.execute("SELECT chunk_data FROM {} WHERE id=?".format(
            self._matrix_table[matrix_id]), (chunk_id,))
        row = cur.fetchone()
        if row is None:
            raise ValueError('ID %s not found.' % chunk_id)
        arr = _load_chunk_content(row[0])
        self.chunk_cache[matrix_id, chunk_id] = arr
        return arr

    def _get_subword_vector_by_ids_dense(self, subword_ids):
        vector = np.zeros((self.metadata['dim'],), dtype='float32')
        chunk = self.metadata['chunk_size']
        last_chunk_id = None
        last_chunk = None
        for subword_id in subword_ids:
            chunk_id, chunk_num = divmod(subword_id, chunk)
            if last_chunk_id == chunk_id:
                chunk_arr = last_chunk
            else:
                chunk_arr = self._get_matrix_chunk(1, chunk_id)
                last_chunk = chunk_arr
                last_chunk_id = chunk_id
            vector += chunk_arr[chunk_num]
        vector /= len(subword_ids)
        return vector

    def _load_input_quant(self):
        if self._input_quant:
            return
        cur = self.db.cursor()
        cur.execute("SELECT key, data FROM input_quant")
        for key, data in cur:
            if key in ('params', 'pq_params', 'npq_params'):
                self._input_quant[key] = json.loads(data.decode('utf-8'))
            else:
                self._input_quant[key] = _load_chunk_content(data)

    def _pq_get_centroids_idx(self, params, m, i):
        ksub = 1 << 8
        dsub = params['dsub']
        if m == params['nsubq'] - 1:
            return m * ksub * dsub + i * params['lastdsub']
        return (m * ksub + i) * dsub

    def _add_subword_vector_by_id_quant(self, vector, subword_id):
        self._load_input_quant()
        if self._input_quant['params']['qnorm']:
            npq_centroids = self._input_quant['npq_centroids']
            npq_params = self._input_quant['npq_params']
            norm_code = self._input_quant['norm_codes'][subword_id]
            idx = self._pq_get_centroids_idx(npq_params, 0, norm_code)
            norm = npq_centroids[idx]
        else:
            norm = np.array(1.0, dtype='float32')
        codes = self._input_quant['codes']
        params = self._input_quant['pq_params']
        nsubq = params['nsubq']
        code_idx = nsubq * subword_id
        dsub = params['dsub']
        d = dsub
        pq_centroids = self._input_quant['pq_centroids']
        pq_params = self._input_quant['pq_params']
        for m in range(nsubq):
            c = self._pq_get_centroids_idx(pq_params, m, codes[code_idx + m])
            if m == nsubq - 1:
                d = params['lastdsub']
            for n in range(d):
                vector[m * dsub + n] += norm * pq_centroids[c + n]
        return vector

    def get_subword_vector(self, word):
        """Get the word vector representation by composing subwords."""
        subword_ids = self.get_subwords(word)[1]
        if not self.metadata.get('quant_input'):
            subword_ids.sort()
            return self._get_subword_vector_by_ids_dense(subword_ids)
        result = np.zeros(self.metadata['dim'], dtype='float32')
        for subword_id in subword_ids:
            self._add_subword_vector_by_id_quant(result, subword_id)
        return result

    def get_word_vector(self, word):
        """Get the vector representation of word (including out-of-vocabulary words)."""
        vec = self.get_dict_word_vector(word)
        if vec is not None:
            return vec
        return self.get_subword_vector(word)

    def get_sentence_vector(self, text):
        """
        Given a string, get a single vector represenation. This function
        assumes to be given a single line of text. We split words on
        whitespace (space, newline, tab, vertical tab) and the control
        characters carriage return, formfeed and the null character.
        """
        if text.find('\n') != -1:
            raise ValueError(
                "predict processes one line at a time (remove \'\\n\')"
            )
        text += "\n"
        dim = self.get_dimension()
        #b = fasttext.Vector(dim)
        #self.f.getSentenceVector(b, text)
        #return np.array(b)

    def get_nearest_neighbors(self, word, k=10, on_unicode_error='strict'):
        word_vec = self.get_word_vector(word)
        return self.get_vector_nearest_neighbors(word_vec, k, on_unicode_error)

    def get_vector_nearest_neighbors(self, word_vec, k=10, on_unicode_error='strict'):
        ulp = np.nextafter(1, 2) - 1
        chunk_size = int(self.metadata['chunk_size'])
        chunk_count = -(self.metadata['nwords'] // -chunk_size)
        k_per_chunk = -(k // -chunk_count)
        nn_items = []
        cur = self.db.cursor()
        cur.execute("SELECT id, chunk_data FROM word_vectors ORDER BY id")
        for cid, cdata in cur:
            matrix = _load_chunk_content(cdata)
            distances = np.linalg.norm(matrix - word_vec, axis=1)
            nearest_neighbor_ids = distances.argsort()[:k_per_chunk]
            for chunk_num in nearest_neighbor_ids:
                chunk_num = int(chunk_num)
                distance = distances[chunk_num]
                if distance < ulp:
                    continue
                heapq.heappush(nn_items, (distance, cid * chunk_size + chunk_num))
        k_items = heapq.nsmallest(k, nn_items)
        qs = ','.join(('?',) * len(k_items))
        vals = [x[1] for x in k_items]
        cur.execute("SELECT id, word FROM words WHERE id IN ({})".format(qs), vals)
        words = dict(cur)
        result = [(distance, words[wid]) for distance, wid in k_items]
        return result

    def get_analogies(self, wordA, wordB, wordC, k=10,
                      on_unicode_error='strict'):
        vec_a = self.get_word_vector(wordA)
        vec_b = self.get_word_vector(wordB)
        vec_c = self.get_word_vector(wordC)
        query = (
            (vec_a / (np.linalg.norm(vec_a) + 1e-8)) -
            (vec_b / (np.linalg.norm(vec_b) + 1e-8)) +
            (vec_c / (np.linalg.norm(vec_c) + 1e-8))
        )
        return self.get_vector_nearest_neighbors(query, k, on_unicode_error)

    def get_word_id(self, word):
        """
        Given a word, get the word id within the dictionary.
        Returns -1 if word is not in the dictionary.
        """
        word_row = self._words.get(word)
        if word_row:
            return word_row[0]
        if self._words_loaded:
            return -1
        cur = self.db.cursor()
        cur.execute("""
            SELECT w.id, w.freq FROM words w WHERE w.word=?
        """, (word,))
        row = cur.fetchone()
        if row is None:
            return -1
        self._words[word] = tuple(row)
        return row[0]

    def get_label_id(self, label):
        """
        Given a label, get the label id within the dictionary.
        Returns -1 if label is not in the dictionary.
        """
        return self.f.getLabelId(label)

    def get_subword_id(self, subword):
        """
        Given a subword, return the index (within input matrix) it hashes to.
        """
        return self.f.getSubwordId(subword)

    def get_subwords(self, word, on_unicode_error='strict'):
        """
        Given a word, get the subwords and their indicies.
        """
        subwords = []
        subword_ids = []
        word_id = self.get_word_id(word)
        if word_id != -1:
            subwords.append(word)
            subword_ids.append(word_id)
        nwords = self.metadata['nwords']
        for subword, sw_hash in zip(*self.split_subword(word)):
            subwords.append(subword)
            subword_ids.append(nwords + sw_hash)
        return subwords, subword_ids

    def get_input_vector(self, ind):
        """
        Given an index, get the corresponding vector of the Input Matrix.
        """
        chunk_id, chunk_num = divmod(ind, self.metadata['chunk_size'])
        chunk_arr = self._get_matrix_chunk(1, chunk_id)
        return chunk_arr[chunk_num]

    def predict(self, text, k=1, threshold=0.0, on_unicode_error='strict'):
        """
        Given a string, get a list of labels and a list of
        corresponding probabilities. k controls the number
        of returned labels. A choice of 5, will return the 5
        most probable labels. By default this returns only
        the most likely label and probability. threshold filters
        the returned labels by a threshold on probability. A
        choice of 0.5 will return labels with at least 0.5
        probability. k and threshold will be applied together to
        determine the returned labels.

        This function assumes to be given
        a single line of text. We split words on whitespace (space,
        newline, tab, vertical tab) and the control characters carriage
        return, formfeed and the null character.

        If the model is not supervised, this function will throw a ValueError.

        If given a list of strings, it will return a list of results as usually
        received for a single line of text.
        """

        def check(entry):
            if entry.find('\n') != -1:
                raise ValueError(
                    "predict processes one line at a time (remove \'\\n\')"
                )
            entry += "\n"
            return entry

        if type(text) == list:
            text = [check(entry) for entry in text]
            all_labels, all_probs = self.f.multilinePredict(
                text, k, threshold, on_unicode_error)

            return all_labels, all_probs
        else:
            text = check(text)
            predictions = self.f.predict(text, k, threshold, on_unicode_error)
            if predictions:
                probs, labels = zip(*predictions)
            else:
                probs, labels = ([], ())

            return labels, np.array(probs, copy=False)

    def get_input_matrix(self):
        """
        Get a reference to the full input matrix of a Model. This only
        works if the model is not quantized.
        """
        if self.f.isQuant():
            raise ValueError("Can't get quantized Matrix")
        return np.array(self.f.getInputMatrix())

    def get_output_matrix(self):
        """
        Get a reference to the full output matrix of a Model. This only
        works if the model is not quantized.
        """
        if self.f.isQuant():
            raise ValueError("Can't get quantized Matrix")
        return np.array(self.f.getOutputMatrix())

    def get_words(self, include_freq=False, on_unicode_error='strict'):
        """
        Get the entire list of words of the dictionary optionally
        including the frequency of the individual words. This
        does not include any subwords. For that please consult
        the function get_subwords.
        """
        words = []
        freqs = []
        if not self._words_loaded:
            cur = self.db.cursor()
            cur.execute("SELECT id, word, freq FROM words ORDER BY id")
            for wid, word, freq in cur:
                self._words[word] = (wid, freq)
            self._words_loaded = True
        for word, row in self._words.items():
            words.append(word)
            freqs.append(row[1])
        if include_freq:
            return (words, np.array(freqs))
        else:
            return words

    def get_line(self, text, on_unicode_error='strict'):
        """
        Split a line of text into words and labels. Labels must start with
        the prefix used to create the model (__label__ by default).
        """

        def check(entry):
            if entry.find('\n') != -1:
                raise ValueError(
                    "get_line processes one line at a time (remove \'\\n\')"
                )
            entry += "\n"
            return entry

        if type(text) == list:
            text = [check(entry) for entry in text]
            return self.f.multilineGetLine(text, on_unicode_error)
        else:
            text = check(text)
            return self.f.getLine(text, on_unicode_error)

    def save_model(self, path):
        """Save the model to the given path"""
        raise NotImplementedError

    def test(self, path, k=1, threshold=0.0):
        """Evaluate supervised model using file given by path"""
        raise NotImplementedError

    def test_label(self, path, k=1, threshold=0.0):
        """
        Return the precision and recall score for each label.

        The returned value is a dictionary, where the key is the label.
        For example:
        f.test_label(...)
        {'__label__italian-cuisine' : {'precision' : 0.7, 'recall' : 0.74}}
        """
        raise NotImplementedError


    def quantize(
        self,
        input=None,
        qout=False,
        cutoff=0,
        retrain=False,
        epoch=None,
        lr=None,
        thread=None,
        verbose=None,
        dsub=2,
        qnorm=False
    ):
        """
        Quantize the model reducing the size of the model and
        it's memory footprint.
        """
        raise NotImplementedError

    def set_matrices(self, input_matrix, output_matrix):
        """
        Set input and output matrices. This function assumes you know what you
        are doing.
        """
        raise NotImplementedError

    @property
    def words(self):
        if self._words is None:
            self._words = self.get_words()
        return self._words

    @property
    def labels(self):
        if self._labels is None:
            self._labels = self.get_labels()
        return self._labels

    def __getitem__(self, word):
        return self.get_word_vector(word)

    def __contains__(self, word):
        return word in self.words
