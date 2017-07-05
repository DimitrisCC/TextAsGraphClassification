from __future__ import print_function
import timeit
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from tensorflow.contrib import learn
from cnn_classifier import cnn_classifier
from data_helpers import compute_nystroem

tf.flags.DEFINE_string("data_file", "bbcsport/graph", "Data source.")
tf.flags.DEFINE_string("community_detection", "louvain", "Employed community detection algorithm (default: louvain)")
tf.flags.DEFINE_boolean("use_nystroem", True, "Use Nystrom method approximate feature map")
tf.flags.DEFINE_boolean("use_node_labels", False, "Take labels of nodes into account")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 100, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "1", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
if FLAGS.use_node_labels:
    from graph_kernels_labeled import sp_kernel, graphlet_kernel, wl_kernel
else:
    from graph_kernels import sp_kernel, graphlet_kernel, wl_kernel
# Load data

kernels = [wl_kernel, sp_kernel, graphlet_kernel]
num_kernels = len(kernels)

print("Computing feature maps...")
start = time.time()
Q = []

if FLAGS.use_nystroem:
    Q, subgraphs, labels, shapes, time2 = compute_nystroem(FLAGS.data_file, FLAGS.use_node_labels, FLAGS.embedding_dim,
                                                           FLAGS.community_detection, kernels)
else:
    print("Not implemented!!!")

M = np.zeros((shapes[0], shapes[1], len(kernels)))
for idx, k in enumerate(kernels):
    M[:, :, idx] = Q[idx]

Q = M

# Binarize labels
s = pd.Series(labels)
y = pd.get_dummies(s).as_matrix()
end = time.time()
dur = end - start - time2
print(dur)

# Build vocabulary
max_document_length = max([len(x.split(" ")) for x in subgraphs])
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x = np.array(list(vocab_processor.fit_transform(subgraphs)))

# Randomly shuffle data
np.random.seed(None)

kf = KFold(n_splits=10)
kf.shuffle = True
accs = []

for train_index, test_index in kf.split(x):
    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]  # to check error out of bounds
    clf = cnn_classifier(sequence_length=x_train.shape[1], num_classes=y_train.shape[1],
                         vocab_size=len(vocab_processor.vocabulary_), num_kernels=num_kernels, Q=Q, FLAGS=FLAGS)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    acc = accuracy_score(y_pred[0], np.argmax(y_test, axis=1))
    accs.append(acc)
    clf.clear_graph()

    print(acc)
    end = time.time()
    print(end - start)
print("Average accuracy: ", np.mean(accs))
