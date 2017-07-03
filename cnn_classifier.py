from __future__ import print_function
from sklearn.base import BaseEstimator, ClassifierMixin
import tensorflow as tf
import numpy as np
import os
import time
import datetime
from data_helpers import compute_kernel, batch_iter
from cnn import CNN
from tensorflow.contrib import learn
from scipy.sparse.linalg import svds
from sklearn.model_selection import KFold


class cnn_classifier(BaseEstimator, ClassifierMixin):
    """An example of classifier"""

    def __init__(self, sequence_length=0, num_classes=0, vocab_size=0, num_kernels=0, step_size=1e-2, Q=None,
                 FLAGS=None):
        """
        Called when initializing the classifier
        """
        self._estimator_type = "classifier"
        self.Q = tf.constant(Q, dtype=tf.float32, name="input_phi")
        self.sequence_length = sequence_length
        self.num_classes = num_classes
        self.vocab_size = vocab_size
        self.num_kernels = num_kernels
        # Data loading params

        self.FLAGS = FLAGS
        self.FLAGS._parse_flags()

        session_conf = tf.ConfigProto(
            device_count={'GPU': 0},
            allow_soft_placement=self.FLAGS.allow_soft_placement,
            log_device_placement=self.FLAGS.log_device_placement)
        self.sess = tf.Session(config=session_conf)

        self.cnn = CNN(self.Q,
                       sequence_length=sequence_length,
                       num_classes=num_classes,
                       vocab_size=vocab_size,
                       num_kernels=self.num_kernels,
                       embedding_size=self.FLAGS.embedding_dim,
                       filter_sizes=list(map(int, self.FLAGS.filter_sizes.split(","))),
                       num_filters=self.FLAGS.num_filters,
                       l2_reg_lambda=self.FLAGS.l2_reg_lambda)

        # Define Training procedure
        self.optimizer = tf.train.AdagradOptimizer(step_size).minimize(self.cnn.loss)
        self.sess.run(tf.global_variables_initializer())

    def fit(self, X, y=None):
        """
        This should fit classifier. All the "work" should be done here.
        Note: assert is not a good choice here and you should rather
        use try/except blog with exceptions. This is just for short syntax.
        """

        # Generate batches
        batches = batch_iter(
            list(zip(X, y)), self.FLAGS.batch_size, self.FLAGS.num_epochs)

        # Training loop. For each batch...
        for batch in batches:
            x_batch, y_batch = zip(*batch)
            feed_dict = {
                self.cnn.input_x: x_batch,
                self.cnn.input_y: y_batch,
                self.cnn.dropout_keep_prob: self.FLAGS.dropout_keep_prob
            }
            _, loss, accuracy = self.sess.run(
                [self.optimizer, self.cnn.loss, self.cnn.accuracy],
                feed_dict)
            # print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            self.setattr(parameter, value)
        return self

    def clear_graph(self):
        tf.reset_default_graph()

    def predict(self, X):
        feed_dict = {
            self.cnn.input_x: X,
            self.cnn.dropout_keep_prob: 1.0
        }
        predictions = self.sess.run(
            [self.cnn.predictions],
            feed_dict)
        return (predictions)

    def score(self, X, y=None):
        feed_dict = {
            self.cnn.input_x: X,
            self.cnn.input_y: y,
            self.cnn.dropout_keep_prob: 1.0
        }
        loss, accuracy = self.sess.run(
            [self.cnn.loss, self.cnn.accuracy],
            feed_dict)
        return (accuracy)
