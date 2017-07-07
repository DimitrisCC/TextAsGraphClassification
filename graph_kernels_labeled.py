from __future__ import print_function

import copy
from collections import defaultdict

import networkx as nx
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import eigs


def sp_kernel(g1, g2=None):
    if g2 is not None:
        graphs = []
        for g in g1:
            graphs.append(g)
        for g in g2:
            graphs.append(g)
    else:
        graphs = g1

    sp_lengths = []

    for graph in graphs:
        sp_lengths.append(nx.shortest_path_length(graph))

    N = len(graphs)
    all_paths = {}
    sp_counts = {}
    for i in range(N):
        sp_counts[i] = {}
        nodes = graphs[i].nodes()
        for v1 in nodes:
            for v2 in nodes:
                if v2 in sp_lengths[i][v1]:
                    label = tuple(
                        sorted([graphs[i].node[v1]['label'], graphs[i].node[v2]['label']]) + [sp_lengths[i][v1][v2]])
                    if label in sp_counts[i]:
                        sp_counts[i][label] += 1
                    else:
                        sp_counts[i][label] = 1

                    if label not in all_paths:
                        all_paths[label] = len(all_paths)

    phi = np.zeros((N, len(all_paths)))

    for i in range(N):
        for label in sp_counts[i]:
            phi[i, all_paths[label]] = sp_counts[i][label]

    if g2 is not None:
        K = np.dot(phi[:len(g1), :], phi[len(g1):, :].T)
    else:
        K = np.dot(phi, phi.T)

    return K


def graphlet_kernel(g1, g2=None):
    if g2 is not None:
        graphs = []
        for g in g1:
            graphs.append(g)
        for g in g2:
            graphs.append(g)
    else:
        graphs = g1

    labels = {}

    for G in graphs:
        for node in G.nodes():
            if G.node[node]["label"] not in labels:
                labels[G.node[node]["label"]] = len(labels)

    L = len(labels)
    B = 2 * pow(L, 3)

    phi = lil_matrix((len(graphs), B))

    graphlets = {}

    ind = 0
    for G in graphs:
        for node1 in G.nodes():
            for node2 in G.neighbors(node1):
                for node3 in G.neighbors(node2):
                    if node1 != node3:
                        if node3 not in G.neighbors(node1):
                            graphlet = (1, min(G.node[node1]['label'], G.node[node3]['label']), G.node[node2]['label'],
                                        max(G.node[node1]['label'], G.node[node3]['label']))
                            increment = 1.0 / 2.0
                        else:
                            labs = sorted([G.node[node1]['label'], G.node[node2]['label'], G.node[node3]['label']])
                            graphlet = (2, labs[0], labs[1], labs[2])
                            increment = 1.0 / 6.0

                        if graphlet not in graphlets:
                            graphlets[graphlet] = len(graphlets)

                        phi[ind, graphlets[graphlet]] += increment

        ind += 1

    if g2 is not None:
        K = np.dot(phi[:len(g1), :], phi[len(g1):, :].T)
    else:
        K = np.dot(phi, phi.T)

    K = np.asarray(K.todense())
    return K


# Compute Weisfeiler-Lehman subtree kernel
def wl_kernel(g1, g2=None, h=6):
    if g2 is not None:
        graphs = []
        for g in g1:
            graphs.append(g)
        for g in g2:
            graphs.append(g)
    else:
        graphs = g1

    labels = {}
    label_lookup = {}
    label_counter = 0

    N = len(graphs)

    orig_graph_map = {it: {i: defaultdict(lambda: 0) for i in range(N)} for it in range(-1, h)}

    # initial labeling
    ind = 0
    for G in graphs:
        labels[ind] = np.zeros(G.number_of_nodes(), dtype=np.int32)
        node2index = {}
        for node in G.nodes():
            node2index[node] = len(node2index)

        for node in G.nodes():
            label = G.node[node]['label']
            if label not in label_lookup:
                label_lookup[label] = len(label_lookup)

            labels[ind][node2index[node]] = label_lookup[label]
            orig_graph_map[-1][ind][label] = orig_graph_map[-1][ind].get(label, 0) + 1

        ind += 1

    compressed_labels = copy.deepcopy(labels)

    # WL iterations
    for it in range(h):
        unique_labels_per_h = set()
        label_lookup = {}
        ind = 0
        for G in graphs:
            node2index = {}
            for node in G.nodes():
                node2index[node] = len(node2index)

            for node in G.nodes():
                node_label = tuple([labels[ind][node2index[node]]])
                neighbors = G.neighbors(node)
                if len(neighbors) > 0:
                    neighbors_label = tuple([labels[ind][node2index[neigh]] for neigh in neighbors])
                    node_label = str(node_label) + "-" + str(sorted(neighbors_label))
                if not node_label in label_lookup:
                    label_lookup[node_label] = len(label_lookup)

                compressed_labels[ind][node2index[node]] = label_lookup[node_label]
                orig_graph_map[it][ind][node_label] = orig_graph_map[it][ind].get(node_label, 0) + 1

            ind += 1

        print("Number of compressed labels at iteration %s: %s" % (it, len(label_lookup)))
        labels = copy.deepcopy(compressed_labels)

    if g2 is not None:
        K = np.zeros((len(g1), len(g2)))
        for it in range(-1, h):
            for i in range(len(g1)):
                for j in range(len(g2)):
                    common_keys = set(orig_graph_map[it][i].keys()) & set(orig_graph_map[it][len(g1) + j].keys())
                    K[i][j] += sum([orig_graph_map[it][i].get(k, 0) * orig_graph_map[it][len(g1) + j].get(k, 0) for k in
                                    common_keys])
    else:
        K = np.zeros((N, N))
        for it in range(-1, h):
            for i in range(N):
                for j in range(N):
                    common_keys = set(orig_graph_map[it][i].keys()) & set(orig_graph_map[it][j].keys())
                    K[i][j] += sum(
                        [orig_graph_map[it][i].get(k, 0) * orig_graph_map[it][j].get(k, 0) for k in common_keys])

    return K


# Compute Pyramid Match kernel
def pm_kernel(g1, g2=None, L=4, d=6):
    if g2 is not None:
        graphs = []
        for g in g1:
            graphs.append(g)
        for g in g2:
            graphs.append(g)
    else:
        graphs = g1

    N = len(graphs)

    labels = {}

    for G in graphs:
        for node in G.nodes():
            if G.node[node]["label"] not in labels:
                labels[G.node[node]["label"]] = len(labels)

    num_labels = len(labels)

    Us = []
    for G in graphs:
        n = G.number_of_nodes()
        if n == 0:
            Us.append(np.zeros((1, d)))
        else:
            A = nx.adjacency_matrix(G).astype(float)
            if n > d + 1:
                Lambda, U = eigs(A, k=d, ncv=10 * d)
                idx = Lambda.argsort()[::-1]
                U = U[:, idx]
            else:
                Lambda, U = np.linalg.eig(A.todense())
                idx = Lambda.argsort()[::-1]
                U = U[:, idx]
                U = U[:, :d]
            U = np.absolute(U)
            Us.append(U)

    Hs = {}
    for i in range(N):
        G = graphs[i]
        nodes = G.nodes()
        Hs[i] = []
        for j in range(L):
            l = 2 ** j
            D = np.zeros((d * num_labels, l))
            T = np.floor(Us[i] * l)
            T[np.where(T == l)] = l - 1
            for p in range(Us[i].shape[0]):
                if p >= len(nodes):
                    continue
                for q in range(Us[i].shape[1]):
                    D[labels[G.node[nodes[p]]['label']] * d + q, int(T[p, q])] = D[labels[G.node[nodes[p]][
                        'label']] * d + q, int(T[p, q])] + 1

            Hs[i].append(D)

    if g2 is not None:
        K = np.zeros((len(g1), len(g2)))

        for i in range(len(g1)):
            for j in range(len(g2)):
                k = 0
                intersec = np.zeros(L)
                for p in range(L):
                    intersec[p] = np.sum(np.minimum(Hs[i][p], Hs[len(g1) + j][p]))

                k = k + intersec[L - 1]
                for p in range(L - 1):
                    k += (1.0 / (2 ** (L - p - 1))) * (intersec[p] - intersec[p + 1])

                K[i, j] = k
    else:
        K = np.zeros((N, N))

        for i in range(N):
            for j in range(i, N):
                k = 0
                intersec = np.zeros(L)
                for p in range(L):
                    intersec[p] = np.sum(np.minimum(Hs[i][p], Hs[j][p]))

                k = k + intersec[L - 1]
                for p in range(L - 1):
                    k += (1.0 / (2 ** (L - p - 1))) * (intersec[p] - intersec[p + 1])

                K[i, j] = k
                K[j, i] = K[i, j]

    return K
