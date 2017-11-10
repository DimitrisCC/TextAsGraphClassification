from itertools import product
import networkx as nx
import warnings

warnings.filterwarnings("ignore")


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
