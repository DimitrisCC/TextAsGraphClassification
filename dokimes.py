from itertools import product
import networkx as nx
import matplotlib.pyplot as plt
import warnings
import numpy as np
import random
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
from data_helpers import load_embeddings_from_db

warnings.filterwarnings("ignore")


def load_word2vec_model(fname='embeddings/GoogleNews-vectors-negative300.bin.gz', vocab=None):
    model = KeyedVectors.load_word2vec_format(fname=fname, fvocab=vocab, binary=True)
    return model.wv


def load_embeddings_(fname, vocab):
    # Reads word embeddings from disk.
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in range(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            if word in vocab:
                word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
            else:
                f.read(binary_len)
    return word_vecs


G = nx.complete_graph(3)
nx.set_node_attributes(G, name='label', values=G.degree())
H = nx.balanced_tree(2, 2)
nx.set_node_attributes(H, name='label', values=H.degree())

Gp = nx.cartesian_product(G, H)


# nx.draw(G, with_labels=True)
# plt.show()
# nx.draw(H, with_labels=True)
# plt.show()


# nx.draw(Gp, with_labels=True)
# plt.show()


def _dict_product(d1, d2):
    return dict((k, (d1.get(k), d2.get(k))) for k in set(d1) | set(d2))


# Generators for producting graph products
def _node_product(G, H, label_threshold=None):
    for u, v in product(G, H):
        attrs = _dict_product(G.node[u], H.node[v])
        if label_threshold is not None:
            G_labels = nx.get_node_attributes(G, 'label')
            H_labels = nx.get_node_attributes(H, 'label')
            if abs(G_labels[u] - H_labels[v]) < label_threshold:
                new_label = G_labels[u] * H_labels[v]
                attrs['label'] = new_label
                yield ((u, v), attrs)
        else:
            yield ((u, v), attrs)


def _edges_of_product(G, H):
    for u, v, c in G.edges(data=True):
        for x, y, d in H.edges(data=True):
            yield (u, x), (v, y), _dict_product(c, d)


def remove_lone_nodes(G):
    degree = G.degree()
    lone_nodes = [u for u in degree if degree[u] == 0]
    return G.remove_nodes_from(lone_nodes)


def random_walk(G, steps):
    def walk(G, node, prev_node, steps):
        if steps == 0:
            return 0
        steps -= 1
        neighs = G.neighbors(node)
        neighs_attrs = nx.get_node_attributes(G.subgraph(neighs), 'label')
        neighs_attrs = list(neighs_attrs.values())
        if prev_node is not None:
            neighs_attrs[neighs.index(prev_node)] = 0  # we don't want to return to the previous node
        choice = np.random.choice(a=len(neighs), p=neighs_attrs / np.sum(neighs_attrs))
        val = walk(G, node=neighs[choice], prev_node=node, steps=steps)
        return neighs_attrs[choice] + val

    n = 0
    for node_attr in G.nodes(data=True):
        node, attr = node_attr
        num_nodes = G.number_of_nodes(), G.number_of_nodes()
        kernel = np.array((num_nodes, num_nodes))
        value = 0
        for step in range(steps):
            neighs = G.neighbors(node)
            value += (step / steps) * walk(G, node=neighs[0], prev_node=None, steps=steps)
            # kernel[n, ]



GH = nx.Graph()
GH.add_nodes_from(_node_product(G, H, label_threshold=10))
GH.add_edges_from(_edges_of_product(G, H))
# nx.draw(GH, with_labels=True)
# plt.show()
remove_lone_nodes(GH)

attrs = nx.get_node_attributes(GH, 'label')

#
attrs_ = attrs.copy()
for k, v in attrs.items():
    attrs_[k] = np.round(v + np.random.rand() * 10, 2)
nx.set_node_attributes(GH, name='label', values=attrs_)
#
print(nx.get_node_attributes(GH, 'label'))

#random_walk(GH, 2)
print(nx.pagerank(GH))

nx.draw(GH, with_labels=True)
plt.show()
random_walk(GH, 3)
# a = load_embeddings_from_db(vocab=["happy", "priest", "ghost"])
print()
# nx.draw(GH, with_labels=True)
# plt.show()

# model = load_word2vec_model(vocab='dokimh_vocab')
# model2 = load_word2vec_model(vocab='dokimh_vocab2')
# print(cosine_similarity(model['happy'], model2['happy']))
# print(cosine_similarity(model.wv['happy'], model2.wv['happy']))

