import networkx as nx
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")
from itertools import product

G = nx.complete_graph(3)
H = nx.balanced_tree(2, 2)

Gp = nx.cartesian_product(G, H)

nx.draw(G, with_labels=True)
plt.show()
nx.draw(H, with_labels=True)
plt.show()
nx.draw(Gp, with_labels=True)
plt.show()


def _dict_product(d1, d2):
    return dict((k, (d1.get(k), d2.get(k))) for k in set(d1) | set(d2))


# Generators for producting graph products
def _node_product(G, H, label_threshold=None):
    G_labels = nx.get_node_attributes(G, 'label')
    H_labels = nx.get_node_attributes(H, 'label')
    for u, v in product(G, H):
        if label_threshold is not None:
            if abs(G_labels[u] - H_labels[v]) < label_threshold:
                yield ((u, v), _dict_product(G.node[u], H.node[v]))


def _edges_of_product(G, H):
    for u, v, c in G.edges(data=True):
        for x, y, d in H.edges(data=True):
            yield (u, x), (v, y), _dict_product(c, d)


GH = nx.Graph()
GH.add_nodes_from(_node_product(G, H))
GH.add_edges_from(_edges_of_product(G, H))
nx.draw(GH, with_labels=True)
plt.show()
