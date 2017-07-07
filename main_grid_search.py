from __future__ import print_function
import tensorflow as tf
import numpy as np
from data_helpers import compute_kernel, compute_nystroem, batch_iter
from tensorflow.contrib import learn
from scipy.sparse.linalg import svds
from sklearn.model_selection import StratifiedKFold, KFold, GridSearchCV
from cnn_classifier import cnn_classifier
from sklearn.metrics import accuracy_score
import pandas as pd
import time
import codecs
from sklearn.preprocessing import LabelEncoder

data_files = ["ENZYMES", "DD", "IMDB-BINARY", "IMDB-MULTI", "MUTAG", "NCI1", "PROTEINS", "PTC_MR"]
labels_files = [True, True, False, False, True, True, True, True]

batch_sizes = [16, 32, 64, 128]
num_epochs_ar = [50, 100, 150, 200]

tf.flags.DEFINE_string("community_detection", "louvain", "Employed community detection algorithm (default: louvain)")
tf.flags.DEFINE_boolean("use_nystroem", True, "Use Nystrom method approximate feature map")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 100, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "1", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

for counter_file, data_file in enumerate(data_files):

    use_node_labels = labels_files[counter_file]

    if use_node_labels:
        from graph_kernels_labeled import sp_kernel, graphlet_kernel, wl_kernel
    else:
        from graph_kernels import sp_kernel, graphlet_kernel, wl_kernel

    kernels = [wl_kernel, graphlet_kernel]
    num_kernels = len(kernels)
    str_kernels = ""
    for kernel in kernels:
        str_kernels += kernel.__name__[:3]

    accuracies_all = []
    times_all = []

    results_file = codecs.open("results/" + data_file + "_nystroem_" + str_kernels + ".txt", "w", encoding="utf-8")
    results_file.write("Dataset: " + data_file + "\n\n")

    print("Dataset: " + data_file)

    FLAGS = tf.flags.FLAGS
    FLAGS._parse_flags()

    FLAGS.num_kernels = len(kernels)

    for iteration in range(10):

        results_file.write("Iteration: " + str(iteration) + "\n\n")

        # Load data
        print("Computing feature maps...")

        Q = []

        if FLAGS.use_nystroem:
            Q, subgraphs, labels, shapes = compute_nystroem(data_file, use_node_labels, FLAGS.embedding_dim,
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

        # Build vocabulary
        max_document_length = max([len(x.split(" ")) for x in subgraphs])
        vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
        x = np.array(list(vocab_processor.fit_transform(subgraphs)))

        # # Randomly shuffle data
        np.random.seed(None)
        kf = KFold(n_splits=10)
        kf.shuffle = True

        best_mean_acc = 0

        accs = []
        best_acc = 0
        times = []
        fold = 0

        for train_index, test_index in kf.split(x):

            results_file.write("Fold: " + str(fold) + "\n")
            best_acc_fold = 0

            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]

            for batch_size in batch_sizes:
                for num_epochs in num_epochs_ar:

                    FLAGS.batch_size = batch_size
                    FLAGS.num_epochs = num_epochs

                    start_time = time.time()

                    clf = cnn_classifier(sequence_length=x_train.shape[1], num_classes=y_train.shape[1],
                                         vocab_size=len(vocab_processor.vocabulary_), num_kernels=num_kernels, Q=Q,
                                         FLAGS=FLAGS)

                    clf.fit(x_train, y_train)

                    y_pred = clf.predict(x_test)
                    acc = accuracy_score(y_pred[0], np.argmax(y_test, axis=1))

                    clf.clear_graph()

                    elapsed_time = time.time() - start_time
                    times.append(elapsed_time)

                    if acc > best_acc_fold:
                        best_acc_fold = acc

            accs.append(best_acc_fold)
            results_file.write("Best Accuracy at fold: " + str(fold) + ":" + str(best_acc_fold) + "\n\n")
            fold += 1

        mean_acc = np.mean(accs)
        mean_time = np.mean(times)
        std = np.std(accs)
        results_file.write(
            "Mean Accuracy: " + str(mean_acc) + ", Std:" + str(std) + ", Mean time:" + str(mean_time) + "\n\n")
        results_file.write("------------------------------\n\n")

        accuracies_all.append(mean_acc)
        times_all.append(mean_time)

    results_file.write("------------------------------\n\n")
    results_file.write("Time: " + str(np.mean(times_all)) + "\n")
    results_file.write("Mean accuracy: " + str(np.mean(accuracies_all)) + "\n")
    results_file.write("Std: " + str(np.std(accuracies_all)) + "\n")
    results_file.close()
