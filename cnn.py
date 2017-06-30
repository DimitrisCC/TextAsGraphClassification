from __future__ import print_function

import tensorflow as tf
import numpy as np


class CNN(object):
    """
    A CNN for graph classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and multi-layer perceptron.
    """
    def __init__(
      self, input_phi, sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters, num_kernels, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.input_phi = input_phi
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        self.embedded_chars = tf.nn.embedding_lookup(self.input_phi, self.input_x)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        #for kernel in range(num_kernels):
        #x = tf.slice(self.embedded_chars, [0, 0, 0, kernel], [-1, -1, -1, 1])
        #self.embedded_chars_expanded = x
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, num_kernels, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1,1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        #self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool = tf.concat(pooled_outputs, 1)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Densely connected layer
        with tf.name_scope("densely-connectd"):
            W1 = tf.get_variable(
                "W1",
                shape=[num_filters_total, 128],
                initializer=tf.contrib.layers.xavier_initializer())
            b1 = tf.Variable(tf.constant(0.1, shape=[128]), name="b1")
            l2_loss += tf.nn.l2_loss(W1)
            l2_loss += tf.nn.l2_loss(b1)
            self.h_fc1 = tf.nn.relu(tf.matmul(self.h_pool_flat, W1) + b1)
            
        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_fc1, self.dropout_keep_prob)
            
       	# Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W2 = tf.get_variable(
                "W2",
                shape=[128, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b2 = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b2")
            l2_loss += tf.nn.l2_loss(W2)
            l2_loss += tf.nn.l2_loss(b2)
            self.scores = tf.matmul(self.h_drop, W2) + b2
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # CalculateMean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
