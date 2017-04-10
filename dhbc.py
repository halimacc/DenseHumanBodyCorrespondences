from math import ceil
import numpy as np
import tensorflow as tf

class DHBC_FCN:
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

        self.deconv11 = self._deconv_layer(self.deconv10, filter_shape=[5, 5, 16, 96], strides=[1, 4, 4, 1], name='deconv11')
        
        self.pred = self._conv_layer(self.deconv11, filter_shape=[1, 1, 16, 500], strides=[1, 1, 1, 1], name='pred')

    def _max_pool(self, bottom, kernel_size, strides, activation, name):
        with tf.variable_scope(name):
            pool = tf.nn.max_pool(bottom, ksize=kernel_size, strides=strides, padding='SAME', name='pool')

            if activation == 'lrn':
                pool = tf.nn.lrn(pool, name='lrn')

            return pool

    def _conv_layer(self, bottom, filter_shape, strides, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(filter_shape)
            conv = tf.nn.conv2d(bottom, filt, strides, padding='SAME')

            conv_biases = self.get_bias([filter_shape[-1]])
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu

    def _deconv_layer(self, bottom, filter_shape, strides, name):
        with tf.variable_scope(name):
            filt = self.get_deconv_filter(filter_shape)

            in_shape = tf.shape(bottom)
            h = in_shape[1] * strides[1]
            w = in_shape[2] * strides[2]
            new_shape = [in_shape[0], h, w, filter_shape[2]]
            deconv = tf.nn.conv2d_transpose(bottom, filt, tf.stack(new_shape), strides, padding='SAME')
            
            deconv_biases = self.get_bias([filter_shape[-2]])
            bias = tf.nn.bias_add(deconv, deconv_biases)

            relu = tf.nn.relu(bias)
            return relu

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
        var = tf.get_variable(name="weights", initializer=init, shape=weights.shape)
        return var

    def get_conv_filter(self, f_shape):
        init = tf.truncated_normal_initializer(dtype=tf.float32, stddev=1e-1)
        var = tf.get_variable(name="weights", initializer=init, shape=f_shape)
        return var

    def get_bias(self, shape):
        init = tf.constant_initializer(value=0, dtype=tf.float32)
        var = tf.get_variable(name="biases", initializer=init, shape=shape)
        return var