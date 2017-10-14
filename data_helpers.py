from __future__ import print_function
import random
import timeit
from collections import Counter
import networkx as nx
import igraph as ig
import numpy as np
import sys, pickle, logging
import copy, time
from nystroem import Nystroem
import graph_of_words as gow

np.random.seed(None)


def load_data(ds_name, use_node_labels, data_type='text'):
    Gs = []
    labels = []
    cats = {}

    if data_type == 'text':
        # encodings
        if ds_name == 'bbc':
            cats = {
                'business': 1,
                'entertainment': 2,
                'politics': 3,
                'sport': 4,
                'tech': 5
            }
        elif ds_name == 'bbcsport':
            cats = {
                'athletics': 1,
                'cricket': 2,
                'football': 3,
                'rugby': 4,
                'tennis': 5
            }

        Gs, labels = gow.docs_to_networkx(ds_name, cats)

    else:
        node2graph = {}
        ds_name_ = ds_name.split('/')[0]

        with open("./datasets/%s/%s_graph_indicator.txt" % (ds_name, ds_name_), "r") as f:
            c = 1
            for line in f:
                node2graph[c] = int(line[:-1])
                if not node2graph[c] == len(Gs):
                    Gs.append(nx.Graph())
                Gs[-1].add_node(c)
                c += 1

        with open("./datasets/%s/%s_A.txt" % (ds_name, ds_name_), "r") as f:
            for line in f:
                edge = line[:-1].split(",")
                edge[1] = edge[1].replace(" ", "")
                Gs[node2graph[int(edge[0])] - 1].add_edge(int(edge[0]), int(edge[1]))

        if use_node_labels:
            with open("./datasets/%s/%s_node_labels.txt" % (ds_name, ds_name_), "r") as f:
                c = 1
                for line in f:
                    node_label = int(line[:-1])
                    Gs[node2graph[c] - 1].node[c]['label'] = node_label
                    c += 1

        with open("./datasets/%s/%s_graph_labels.txt" % (ds_name, ds_name_), "r") as f:
            for line in f:
                if line is None or line[:-1] == '':
                    continue
                labels.append(int(line[:-1]))

    labels = np.array(labels, dtype=np.float)
    return Gs, labels


def networkx_to_igraph(G):
    mapping = dict(zip(G.nodes(), range(G.number_of_nodes())))
    reverse_mapping = dict(zip(range(G.number_of_nodes()), G.nodes()))
    G = nx.relabel_nodes(G, mapping)
    G_ig = ig.Graph(len(G), list(zip(*list(zip(*nx.to_edgelist(G)))[:2])))
    return G_ig, reverse_mapping


def neighbors_community(G):
    communities = []
    for v in G.nodes():
        communities.append(G.neighbors(v))
    return communities


def neighbors2_community(G):
    communities = []
    for v in G.nodes():
        neighs = G.neighbors(v)
        communities.append(v)
        for n in neighs:
            communities.append(n)

    return communities


def community_detection(G_networkx, community_detection_method):
    G, reverse_mapping = networkx_to_igraph(G_networkx)

    if community_detection_method == "neighbors":
        communities = neighbors_community(G_networkx)
        return communities
    if community_detection_method == "neighbors2":
        communities = neighbors2_community(G_networkx)
        return communities
    if community_detection_method == "eigenvector":
        c = G.community_leading_eigenvector()
    elif community_detection_method == "infomap":
        c = G.community_infomap()
    elif community_detection_method == "fastgreedy":
        c = G.community_fastgreedy().as_clustering()
    elif community_detection_method == "label_propagation":
        c = G.community_label_propagation()
    elif community_detection_method == "louvain":
        c = G.community_multilevel()
    elif community_detection_method == "spinglass":
        c = G.community_spinglass()
    elif community_detection_method == "walktrap":
        c = G.community_walktrap().as_clustering()
    elif community_detection_method == 'multilevel':
        c = G.community_multilevel()
    elif community_detection_method == 'edge_betweenness':
        c = G.community_edge_betweenness().as_clustering()
    else:
        c = []

    communities = []
    for i in range(len(c)):
        community = []
        for j in range(len(c[i])):
            community.append(reverse_mapping[G.vs[c[i][j]].index])

        communities.append(community)

    return communities


def compute_communities(graphs, use_node_labels, community_detection_method):
    communities = []
    subgraphs = []
    counter = 0
    for G in graphs:
        c = community_detection(G, community_detection_method)
        subgraph = []
        for i in range(len(c)):
            communities.append(G.subgraph(c[i]))
            subgraph.append(counter)
            counter += 1

        subgraphs.append(' '.join(str(s) for s in subgraph))
    return communities, subgraphs


def compute_kernel(ds_name, use_node_labels, community_detection_method):
    graphs, labels = load_data(ds_name, use_node_labels)
    communities, subgraphs = compute_communities(graphs, use_node_labels, community_detection_method)
    if use_node_labels:
        from graph_kernels_labeled import sp_kernel, graphlet_kernel, wl_kernel, pm_kernel
    else:
        from graph_kernels import sp_kernel, graphlet_kernel, wl_kernel, pm_kernel
    K = wl_kernel(communities)
    return K, subgraphs, labels


def generate_data(ds_name):
    graphs = []
    labels = []
    ni = np.random.binomial(n=300, p=0.5, size=100000)
    ei = np.random.binomial(n=3, p=0.3, size=100000)

    for i in range(2000):
        n = random.uniform(130, 180)
        e = random.uniform(0.15, 0.45)
        g1 = nx.fast_gnp_random_graph(n, e)
        graphs.append(g1)
        labels.append(1)
        n = random.uniform(130, 180)
        e = random.uniform(0.15, 0.45)

        n = random.uniform(130, 180)
        e = random.uniform(0.15, 0.45)

    pass


def compute_nystroem(ds_name, use_node_labels, embedding_dim, community_detection_method, kernels):
    start = time.time()
    graphs, labels = load_data(ds_name, use_node_labels)
    # graphs, labels = generate_data(ds_name)
    end = time.time()
    time2 = end - start
    communities, subgraphs = compute_communities(graphs, use_node_labels, community_detection_method)
    Q = []
    for idx, k in enumerate(kernels):
        model = Nystroem(k, n_components=embedding_dim)
        model.fit(communities)
        print(len(communities))
        Q_t = model.transform(communities)
        Q_t = np.vstack([np.zeros(embedding_dim), Q_t])
        Q.append(Q_t)

    return Q, subgraphs, labels, Q_t.shape, time2


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
