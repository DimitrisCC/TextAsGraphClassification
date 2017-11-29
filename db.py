import sqlite3 as sql
from gensim.models import KeyedVectors

con = None


def connect():
    global con
    con = sql.connect('embeddings.db')
    return con


def create():
    global con
    with con:
        cur = con.cursor()
        cur.execute(
            "CREATE TABLE IF NOT EXISTS Embeddings(word TEXT, embedding TEXT, PRIMARY KEY (word) ON CONFLICT IGNORE)")


def insert(words, embeddings):
    global con
    with con:
        cur = con.cursor()
        embed_arr = []
        for i in range(len(words)):
            for j in range(len(embeddings)):
                embed_arr.append([words[i], embeddings[i]])
        cur.executemany("INSERT OR IGNORE INTO Tweets VALUES(?, ?)", embed_arr)


def select(word):
    global con
    with con:
        cur = con.cursor()
        cur.execute("SELECT %s FROM Embeddings" % word)
        return cur.fetchall()


def load_word2vec_model(fname='embeddings/GoogleNews-vectors-negative300.bin.gz', vocab=None):
    model = KeyedVectors.load_word2vec_format(fname=fname, fvocab=vocab, binary=True)
    return model.wv
