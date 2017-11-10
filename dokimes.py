import networkx as nx
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")
from itertools import product

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
    for node_attr in G.nodes(data=True):
        node, attr = node_attr
        for step in range(steps):
            neighs = G.neighbors(node)


GH = nx.Graph()
GH.add_nodes_from(_node_product(G, H, label_threshold=10))
GH.add_edges_from(_edges_of_product(G, H))
# nx.draw(GH, with_labels=True)
# plt.show()
remove_lone_nodes(GH)
print(nx.get_node_attributes(GH, 'label'))
nx.draw(GH, with_labels=True)
plt.show()

random_walk(GH, 2)
print(nx.pagerank(GH))
