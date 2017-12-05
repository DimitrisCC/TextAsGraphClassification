import sqlite3 as sql
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity

con = None


def connect():
    global con
    con = sql.connect('embeddings/word2vec.db')
    return con


def create():
    global con
    with con:
        cur = con.cursor()
        cur.execute(
            "CREATE TABLE IF NOT EXISTS Embeddings(word TEXT, embedding TEXT, PRIMARY KEY (word) ON CONFLICT IGNORE)")


def insert(model):
    global con
    with con:
        cur = con.cursor()
        print('insert mesa')
        cur.executemany("INSERT OR IGNORE INTO Embeddings VALUES(?, ?)", model)


def select(word=None):
    global con
    with con:
        cur = con.cursor()
        query = "SELECT word, embedding FROM Embeddings"
        if word is not None:
            query += " WHERE word" % word
        cur.execute(query)
        data = cur.fetchall()
        model = {}
        for row in data:
            model[row[0]] = list(map(float, row[1].split(',')))
        return model


def load_word2vec_model(fname='embeddings/GoogleNews-vectors-negative300.bin.gz', vocab=None):
    model = KeyedVectors.load_word2vec_format(fname=fname, fvocab=vocab, binary=True)
    return model.wv


def main():
    from data_helpers import load_embeddings
    model = load_embeddings(fvocab=None, use_db=False, as_dict=False)
    strmodel = []
    for word in model.wv.vocab:
        emb = ",".join(map(str, model.wv[word]))
        strmodel.append([word, emb])
    print('ok')
    connect()
    print('con')
    create()
    print('create')
    insert(strmodel)


if __name__ == "__main__":
    main()
