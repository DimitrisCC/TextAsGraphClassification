import itertools
import re
from sklearn import preprocessing
import os
import numpy as np
import networkx as nx
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import pprintpp as pp

tags = ['NN', 'NNP', 'NNS', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']


def clean_terms(terms, stopwords=None, lemmatize=None, stem=None, only_N_V=None):
    if stopwords is not None:
        terms = [t for t in terms if t not in stopwords]
    if only_N_V is not None:  # include only nouns and verbs
        tagged = nltk.pos_tag(terms)
        terms = [t for t, pos in tagged if pos in tags]
    if lemmatize is not None:
        lem = WordNetLemmatizer()
        terms = [lem.lemmatize(t) for t in terms]
    if stem is not None:
        stem = PorterStemmer()
        terms = [stem.stem(t) for t in terms]
    return terms


def extract_terms_from_file(file_location, stopwords=None, lemmatize=None, stem=None, only_N_V=None):
    with open(file_location, 'r', encoding='iso-8859-1') as doc:
        terms = []
        for line in doc:
            terms.extend(re.compile('\w+').findall(line.lower()))

        # terms = re.compile('\w+').findall(doc
        #                                   .read()
        #                                   .replace('\n', '')
        #                                   .lower())
        return clean_terms(terms, stopwords, lemmatize, stem, only_N_V)


def extract_terms_from_sentence(sentence, stopwords=None, lemmatize=None, stem=None, only_N_V=None):
    terms = re.compile('\w+').findall(sentence.lower())
    return clean_terms(terms, stopwords, lemmatize, stem, only_N_V)


def terms_to_graph(terms, w):  # terms=list w=window size

    from_to = {}

    # create initial graph (first w terms)
    terms_temp = terms[0:w]
    indexes = list(itertools.combinations(range(w), r=2))

    new_edges = list()

    for i in range(len(indexes)):
        new_edges.append(" ".join(list(terms_temp[i] for i in indexes[i])))

    for i in range(0, len(new_edges)):
        from_to[new_edges[i].split()[0], new_edges[i].split()[1]] = 1

    # then iterate over the remaining terms
    for i in range(w, len(terms)):
        # term to consider
        considered_term = terms[i]
        # all terms within sliding window
        terms_temp = terms[(i - w + 1):(i + 1)]

        # edges to try
        candidate_edges = list()
        for p in range(w - 1):
            candidate_edges.append((terms_temp[p], considered_term))

        for try_edge in candidate_edges:

            # if not self-edge
            if try_edge[1] != try_edge[0]:

                boolean1 = (try_edge[0], try_edge[1]) in from_to
                boolean2 = (try_edge[1], try_edge[0]) in from_to

                # if edge has already been seen, update its weight
                if boolean1:
                    from_to[try_edge[0], try_edge[1]] += 1

                elif boolean2:
                    from_to[try_edge[1], try_edge[0]] += 1

                # if edge has never been seen, create it and assign it a unit weight
                else:
                    from_to[try_edge] = 1

    return from_to


def graph_to_networkx(graph, name=None):
    G = nx.Graph()
    G.name = name
    for edge, weight in graph.items():
        G.add_edge(*edge, weight=weight)
    return G


def docs_to_networkx(dataset, cats, window_size=2):
    ds = './datasets/%s/' % dataset
    Gs = []
    labels = []
    type_ = 2

    for doc in os.listdir(ds):
        if 'train.txt' in doc:
            type_ = 1

    if type_ == 1:
        with open(ds + '/train.txt', 'r', encoding='iso-8859-1') as doc:
            dc = 1
            for line in doc:
                label = line[0]
                labels.append(label)
                terms = extract_terms_from_sentence(line[1:],
                                                    stopwords=stopwords.words('english'),
                                                    lemmatize=True,
                                                    stem=True,
                                                    only_N_V=True)
                graph = terms_to_graph(terms, window_size)
                G = graph_to_networkx(graph, name=label + '_' + str(dc))
                # G = nx.convert_node_labels_to_integers(G, first_label=1, label_attribute='label')
                nx.set_node_attributes(G, 'label', dict(zip(G.nodes(), G.nodes())))
                Gs.append(G)
                dc += 1
    else:
        for cat in cats.keys():
            for doc in os.listdir(ds + cat):
                terms = extract_terms_from_file(ds + cat + '/' + doc,
                                                stopwords=stopwords.words('english'),
                                                lemmatize=True,
                                                stem=True,
                                                only_N_V=True)
                graph = terms_to_graph(terms, window_size)
                G = graph_to_networkx(graph, name=cat + doc.split('.')[0])
                # G = nx.convert_node_labels_to_integers(G, first_label=1, label_attribute='label')
                nx.set_node_attributes(G, 'label', dict(zip(G.nodes(), G.nodes())))
                Gs.append(G)
                labels.append(cats[cat])
    return Gs, labels


# needs fix or discard
def produce_graph_files(dataset, cats):
    def get_text_to_write(arr):
        to_write = ""
        for i in range(len(arr)):
            if i % 2 == 0:
                to_write += str(arr[i]) + ', '
            else:
                to_write += str(arr[i]) + '\n'
        return to_write

    # data to be written in files
    A_ = []
    terms_ = []
    graph_indicator_ = []
    graph_labels_ = []
    graph_labels_str_ = []
    edge_labels_ = []

    ds = './datasets/%s/' % dataset

    # preprocessing
    gid = 1  # graph indicator
    for cat in cats.keys():
        for doc in os.listdir(ds + cat):
            terms = extract_terms_from_file(ds + cat + '/' + doc)
            graph = terms_to_graph(terms, 2)

            termlist = [[*x] for x in graph.keys()]
            termlist = np.array(termlist).flatten().tolist()

            # just the terms
            terms_.extend(termlist)
            # append the graph indicator to each term to differentiate the same terms of different documents and classes
            # this is useful later when label encoding is performed
            A_.extend([t + doc for t in termlist])
            # each edge has a weight
            edge_labels_.extend([w for w in graph.values()])
            # gid for each node in the graph - unique term
            graph_indicator_.extend([gid for _ in range(len(set(terms)))])
            # label for each graph as integer
            graph_labels_.append(cats[cat])
            # label for each graph as string
            graph_labels_str_.append(cat)

            gid += 1

    le = preprocessing.LabelEncoder()
    A_ = le.fit_transform(A_) + 1
    A_ = A_.tolist()
    A_ = get_text_to_write(A_)
    terms_ = get_text_to_write(terms_)
    edge_labels_ = '\n'.join([str(x) for x in edge_labels_])
    graph_indicator_ = '\n'.join([str(x) for x in graph_indicator_])
    graph_labels_ = '\n'.join([str(x) for x in graph_labels_])
    graph_labels_str_ = '\n'.join(graph_labels_str_)

    # files
    gds = ds + 'graph/'
    os.makedirs(os.path.dirname(gds), exist_ok=True)
    with open(gds + dataset + '_terms.txt', 'w') as terms_file:
        terms_file.write(terms_)
    with open(gds + dataset + '_A.txt', 'w') as A_file:
        A_file.write(A_)
    with open(gds + dataset + '_edge_labels.txt', 'w') as edge_labels_file:
        edge_labels_file.write(edge_labels_)
    with open(gds + dataset + '_graph_indicator.txt', 'w') as graph_indicator_file:
        graph_indicator_file.write(graph_indicator_)
    with open(gds + dataset + '_graph_labels.txt', 'w') as graph_labels_file:
        graph_labels_file.write(graph_labels_)
    with open(gds + dataset + '_graph_labels_str.txt', 'w') as graph_labels_str_file:
        graph_labels_str_file.write(graph_labels_str_)


if __name__ == "__main__":
    # encodings
    cats = {
        'business': 1,
        'entertainment': 2,
        'politics': 3,
        'sport': 4,
        'tech': 5
    }

    scats = {
        'athletics': 1,
        'cricket': 2,
        'football': 3,
        'rugby': 4,
        'tennis': 5
    }

    # produce_graph_files('bbcsport', scats)
    # produce_graph_files('bbc', cats)
