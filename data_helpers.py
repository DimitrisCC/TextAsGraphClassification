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
from gensim.models import KeyedVectors

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


def load_embeddings(fname='embeddings/GoogleNews-vectors-negative300.bin.gz', fvocab=None, as_dict=True):
    model = KeyedVectors.load_word2vec_format(fname=fname, fvocab=fvocab, binary=True)
    if as_dict:
        word_vecs = {}
        for word in model.wv.vocab:
            vec = model.wv[word]
            word_vecs[word] = vec
            return word_vecs
    else:
        return model


def add_unknown_words(word_vecs, vocab, k=300):
    # For words not existing in the pretrained dataset, create a separate word vector.
    for word in vocab:
        if word not in word_vecs:
            word_vecs[word] = np.random.uniform(-0.25, 0.25, k)


def networkx_to_igraph(G):
    mapping = dict(zip(G.nodes(), range(G.number_of_nodes())))
    reverse_mapping = dict(zip(range(G.number_of_nodes()), G.nodes()))
    G = nx.relabel_nodes(G, mapping)
    G_ig = ig.Graph(len(G), list(zip(*list(zip(*nx.to_edgelist(G)))[:2])))
    return G_ig, reverse_mapping


def neighbors_community(G):
    communities = []
    for v in G.nodes():
        community = G.neighbors(v)
        community.append(v)
        communities.append(community)
    return communities


def neighbors2_community(G, remove_duplicates=True, use_kcore=False):
    Gc = None
    if use_kcore:
        Gc = G.copy()
        Gc.remove_edges_from(Gc.selfloop_edges())
        Gc = nx.k_core(Gc, 3)
        # Gc = [cl for cl in nx.find_cliques(G)]
    else:
        Gc = G

    communities = set()

    for v in Gc.nodes():
        neighs = G.neighbors(v)
        community = []
        for n in neighs:
            community.append(n)
            neighs2 = G.neighbors(n)
            community.extend(neighs2)
        if remove_duplicates:
            community = list(set(community))
        communities.add(tuple(community))

    communities = list(map(list, communities))  # Convert tuples back into lists
    return communities


def community_detection(G_networkx, community_detection_method):
    if community_detection_method == "neighbors":
        communities = neighbors_community(G_networkx)
        return communities
    if community_detection_method == "neighbors2":
        communities = neighbors2_community(G_networkx)
        return communities

    G, reverse_mapping = networkx_to_igraph(G_networkx)

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
    coms = []
    for G in graphs:
        c = community_detection(G, community_detection_method)
        coms.append(len(c))
        subgraph = []
        for i in range(len(c)):
            communities.append(G.subgraph(c[i]))
            subgraph.append(counter)
            counter += 1

        subgraphs.append(' '.join(str(s) for s in subgraph))

    print("Average communities: ", np.mean(coms))
    return communities, subgraphs


def compute_embeddings(ds_name, use_node_labels, community_detection_method, kernels):
    graphs, labels = load_data(ds_name, use_node_labels)
    communities, subgraphs = compute_communities(graphs, use_node_labels, community_detection_method)

    Q = []
    for idx, k in enumerate(kernels):
        Q_t = k(communities, explicit=True)
        dim = Q_t.shape[1]
        Q_t = np.vstack([np.zeros(dim), Q_t])
        Q.append(Q_t)

    return Q, subgraphs, labels, Q_t.shape


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

    print("Total communities: ", len(communities))
    lens = []
    for community in communities:
        lens.append(community.number_of_nodes())

    print("Average size: ", np.mean(lens))

    Q = []
    for idx, k in enumerate(kernels):
        model = Nystroem(k, n_components=embedding_dim)
        model.fit(communities)
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


            # def create_train_test_loaders(Q, x_train, x_test, y_train, y_test, batch_size):
            #     num_kernels = Q.shape[2]
            #     max_document_length = x_train.shape[1]
            #     dim = Q.shape[1]
            #
            #     my_x = []
            #     for i in range(x_train.shape[0]):
            #         temp = np.zeros((1, num_kernels, max_document_length, dim))
            #         for j in range(num_kernels):
            #             for k in range(x_train.shape[1]):
            #                 temp[0, j, k, :] = Q[x_train[i, k], :, j].squeeze()
            #         my_x.append(temp)
            #
            #     if torch.cuda.is_available():
            #         tensor_x = torch.stack([torch.cuda.FloatTensor(i) for i in my_x])  # transform to torch tensors
            #         tensor_y = torch.cuda.LongTensor(y_train.tolist())
            #     else:
            #         tensor_x = torch.stack([torch.Tensor(i) for i in my_x])  # transform to torch tensors
            #         tensor_y = torch.from_numpy(np.asarray(y_train, dtype=np.int64))
            #
            #     train_dataset = utils.TensorDataset(tensor_x, tensor_y)
            #     train_loader = utils.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            #
            #     my_x = []
            #     for i in range(x_test.shape[0]):
            #         temp = np.zeros((1, num_kernels, max_document_length, dim))
            #         for j in range(num_kernels):
            #             for k in range(x_test.shape[1]):
            #                 temp[0, j, k, :] = Q[x_test[i, k], :, j].squeeze()
            #         my_x.append(temp)
            #
            #     if torch.cuda.is_available():
            #         tensor_x = torch.stack([torch.cuda.FloatTensor(i) for i in my_x])  # transform to torch tensors
            #         tensor_y = torch.cuda.LongTensor(y_test.tolist())
            #     else:
            #         tensor_x = torch.stack([torch.Tensor(i) for i in my_x])  # transform to torch tensors
            #         tensor_y = torch.from_numpy(np.asarray(y_test, dtype=np.int64))
            #
            #     test_dataset = utils.TensorDataset(tensor_x, tensor_y)
            #     test_loader = utils.DataLoader(test_dataset, batch_size=1, shuffle=False)
            #
            #     return train_loader, test_loader
