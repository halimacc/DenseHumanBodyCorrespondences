import os
import logging
from math import ceil
import sys
from loss import loss

import numpy as np
import tensorflow as tf


class FCNDHBC:

    def __init__(self):
        self.wd = 5e-4

    def build(self, depth, train=False):
        self.conv1 = self._conv_layer(depth, filter_shape=[11, 11, 1, 96], strides=[1, 4, 4, 1], name="conv1")
        
        self.pool1 = self._max_pool(self.conv1, kernel_size=[1, 3, 3, 1], strides=[1, 2, 2, 1], activation='lrn', name='pool1')

        self.conv2 = self._conv_layer(self.pool1, filter_shape=[5, 5, 96, 256], strides=[1, 1, 1, 1], name="conv2")
        self.pool2 = self._max_pool(self.conv2, kernel_size=[1, 3, 3, 1], strides=[1, 2, 2, 1], activation='lrn', name='pool2')

        self.conv3 = self._conv_layer(self.pool2, filter_shape=[3, 3, 256, 384], strides=[1, 1, 1, 1], name="conv3")

        self.conv4 = self._conv_layer(self.conv3, filter_shape=[3, 3, 384, 384], strides=[1, 1, 1, 1], name="conv4")

        self.conv5 = self._conv_layer(self.conv4, filter_shape=[3, 3, 384, 256], strides=[1, 1, 1, 1], name="conv5")
        self.pool5 = self._max_pool(self.conv5, kernel_size=[1, 3, 3, 1], strides=[1, 2, 2, 1], activation='idn', name='pool5')

        self.fc6 = self._conv_layer(self.pool5, filter_shape=[1, 1, 256, 4096], strides=[1, 1, 1, 1], name="fc6")
        if train:
            self.fc6 = tf.nn.dropout(self.fc6, 0.5)

        self.fc7 = self._conv_layer(self.fc6, filter_shape=[1, 1, 4096, 4096], strides=[1, 1, 1, 1], name="fc7")
        if train:
            self.fc7 = tf.nn.dropout(self.fc7, 0.5)

        self.deconv8 = self._deconv_layer(self.fc7, filter_shape=[3, 3, 256, 4096], strides=[1, 2, 2, 1], name='deconv8')

        self.deconv9 = self._deconv_layer(self.deconv8, filter_shape=[3, 3, 256, 256], strides=[1, 2, 2, 1], name='deconv9')

        self.deconv10 = self._deconv_layer(self.deconv9, filter_shape=[3, 3, 96, 256], strides=[1, 2, 2, 1], name='deconv10')

        self.deconv11 = self._deconv_layer(self.deconv10, filter_shape=[3, 3, 16, 96], strides=[1, 4, 4, 1], name='deconv11')

        #self.classify = self._conv_layer(self.deconv11, filter_shape=[1, 1, 16, 500], strides=[1, 1, 1, 1], name='classify')
        #self.pred = tf.nn.softmax(self.classify, dim=3)
        self.pred = self._conv_layer(self.deconv11, filter_shape=[1, 1, 16, 500], strides=[1, 1, 1, 1], name='classify')

    def _max_pool(self, bottom, kernel_size, strides, activation, name):
        with tf.variable_scope(name) as scope:
            pool = tf.nn.max_pool(bottom, ksize=kernel_size, strides=strides, padding='SAME')

            if activation == 'lrn':
                pool = tf.nn.lrn(pool)

            return pool

    def _conv_layer(self, bottom, filter_shape, strides, name):
        with tf.variable_scope(name) as scope:
            filt = self.get_conv_filter(filter_shape)
            conv = tf.nn.conv2d(bottom, filt, strides, padding='SAME')

            conv_biases = self.get_bias([filter_shape[3]])
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            # Add summary to Tensorboard
            #_activation_summary(relu)
            return relu

    def _deconv_layer(self, bottom, filter_shape, strides, name):
        with tf.variable_scope(name) as scope:
            filt = self.get_deconv_filter(filter_shape)
            self._add_wd_and_summary(filt, self.wd, "fc_wlosses")

            in_shape = tf.shape(bottom)

            h = in_shape[1] * strides[1]
            w = in_shape[2] * strides[2]
            new_shape = [in_shape[0], h, w, filter_shape[2]]
            deconv = tf.nn.conv2d_transpose(bottom, filt, tf.stack(new_shape), strides, padding='SAME')

            #_activation_summary(deconv)
            return deconv

    def get_deconv_filter(self, f_shape):
        width = f_shape[0]
        heigh = f_shape[0]
        f = ceil(width / 2.0)
        c = (2 * f - 1 - f % 2) / (2.0 * f)
        bilinear = np.zeros([f_shape[0], f_shape[1]])
        for x in range(width):
            for y in range(heigh):
                value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
                bilinear[x, y] = value
        weights = np.zeros(f_shape)
        for i in range(f_shape[2]):
            weights[:, :, i, i] = bilinear

        init = tf.constant_initializer(value=weights, dtype=tf.float32)
        var = tf.get_variable(name="up_filter", initializer=init, shape=weights.shape)
        return var

    def get_conv_filter(self, f_shape):
        init = tf.truncated_normal_initializer(dtype=tf.float32, stddev=1e-1)
        var = tf.get_variable(name="filter", initializer=init, shape=f_shape)
        if not tf.get_variable_scope().reuse:
            weight_decay = tf.multiply(tf.nn.l2_loss(var), self.wd, name='weight_loss')
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, weight_decay)
        #_variable_summaries(var)
        return var

    def get_bias(self, shape):
        init = tf.constant_initializer(value=0, dtype=tf.float32)
        var = tf.get_variable(name="biases", initializer=init, shape=shape)
        #_variable_summaries(var)
        return var

    def _variable_with_weight_decay(self, shape, stddev, wd, decoder=False):
        """Helper to create an initialized Variable with weight decay.
        Note that the Variable is initialized with a truncated normal
        distribution.
        A weight decay is added only if one is specified.
        Args:
          name: name of the variable
          shape: list of ints
          stddev: standard deviation of a truncated Gaussian
          wd: add L2Loss weight decay multiplied by this float. If None, weight
              decay is not added for this Variable.
        Returns:
          Variable Tensor
        """

        initializer = tf.truncated_normal_initializer(stddev=stddev)
        var = tf.get_variable('weights', shape=shape,
                              initializer=initializer)

        collection_name = tf.GraphKeys.REGULARIZATION_LOSSES
        if wd and (not tf.get_variable_scope().reuse):
            weight_decay = tf.multiply(
                tf.nn.l2_loss(var), wd, name='weight_loss')
            tf.add_to_collection(collection_name, weight_decay)
        _variable_summaries(var)
        return var

    def _add_wd_and_summary(self, var, wd, collection_name=None):
        if collection_name is None:
            collection_name = tf.GraphKeys.REGULARIZATION_LOSSES
        if wd and (not tf.get_variable_scope().reuse):
            weight_decay = tf.multiply(
                tf.nn.l2_loss(var), wd, name='weight_loss')
            tf.add_to_collection(collection_name, weight_decay)
        _variable_summaries(var)
        return var

    def _bias_variable(self, shape, constant=0.0):
        initializer = tf.constant_initializer(constant)
        var = tf.get_variable(name='biases', shape=shape,
                              initializer=initializer)
        _variable_summaries(var)
        return var


def _activation_summary(x):
    """Helper to create summaries for activations.
    Creates a summary that provides a histogram of activations.
    Creates a summary that measure the sparsity of activations.
    Args:
      x: Tensor
    Returns:
      nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = x.op.name
    # tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))


def _variable_summaries(var):
    """Attach a lot of summaries to a Tensor."""
    if not tf.get_variable_scope().reuse:
        name = var.op.name
        logging.info("Creating Summary for: %s" % name)
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.summary.scalar(name + '/mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
            tf.summary.scalar(name + '/sttdev', stddev)
            tf.summary.scalar(name + '/max', tf.reduce_max(var))
            tf.summary.scalar(name + '/min', tf.reduce_min(var))
            tf.summary.histogram(name, var)
