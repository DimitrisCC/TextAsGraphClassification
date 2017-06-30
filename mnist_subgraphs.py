from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
flags = tf.app.flags
FLAGS = flags.FLAGS
import matplotlib.pyplot as plt
# Graphs.

import networkx as nx
# Directories.
plt.interactive(False)
from sklearn.feature_extraction import image
import pickle

import sys, os
sys.path.insert(0, '..')

import numpy as np

def test():
    mnist = input_data.read_data_sets("MINST_data", one_hot=False)
    train_data = mnist.train.images.astype(np.float32)
    fraction=50
    train_labels= mnist.train._labels[:fraction]
    with open('sugbgraphs_labels.pickle', 'wb') as f:
        pickle.dump(train_labels, f)

    test_data = mnist.test.images.astype(np.float32)
    print(train_data.shape)
    patch_size=4
    n_ids=range(patch_size*patch_size)
    A=np.ones((patch_size*patch_size,patch_size*patch_size))
    np.fill_diagonal(A, 0)
    cc = 0
    train=[]

    bins = list(np.linspace(0.0, 1.0, 10))
    for sample in train_data[:fraction]:
        sample=sample.reshape((28, 28))
        sugbg = []
        patches = image.extract_patches_2d(sample, (patch_size, patch_size))
        cc+=1
        for p in patches:
                if np.sum(p) == 0:
                    continue
                G1=nx.from_numpy_matrix(A)
                dictionary = dict(zip(n_ids, np.digitize(p.flatten(), bins)))
                nx.set_node_attributes(G1, 'label', dictionary)
                sugbg.append(G1)
        train.append(sugbg)
        print(cc)

    with open('sugbgraphs_train.pickle', 'wb') as f:
        pickle.dump(train, f)

    del train
    test = []
    for sample in test_data[:5]:
        sample = sample.reshape((28, 28))
        sugbg = []
        patches = image.extract_patches_2d(sample, (patch_size, patch_size))
        for p in patches:
            if np.sum(p) == 0:
                continue

            G1 = nx.from_numpy_matrix(A)
            p = np.histogram(p.flatten(), bins=np.linspace(0.0, 1.0, 10))[0]
            dictionary = dict(zip(n_ids, p))
            nx.set_node_attributes(G1, 'label', dictionary)
            sugbg.append(G1)
        test.append(sugbg)
    with open('sugbgraphs_test.pickle', 'wb') as f:
        pickle.dump(sugbg, f)

if __name__=="__main__":
    test()