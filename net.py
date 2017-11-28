from config import *
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

class DHBC:
    def __init__(self):
        self.feature = None
        self.feat_vars = None
        self.preds = {}
        self.clas_vars = {}
        self.losses = {}
        self.train_ops = {}

    def forward(self, input):
        self.depth = input
        conv = self._conv
        upconv = self._upconv
        maxpool = self._maxpool
        with tf.variable_scope('feature'):
            with tf.variable_scope('encoder'):
                conv1 = conv(self.depth, 96, 11, 4)                         # H/4
                pool1 = maxpool(conv1, 3)                                   # H/8
                conv2 = conv(pool1, 256, 5, 1)                              # H/8
                pool2 = maxpool(conv2, 3)                                   # H/16
                conv3 = conv(pool2, 384, 3, 1)                              # H/16
                conv4 = conv(conv3, 384, 3, 1)                              # H/16
                conv5 = conv(conv4, 256, 3, 1)                              # H/16
                pool5 = maxpool(conv5, 3)                                   # H/32
                conv6 = conv(pool5, 4096, 1, 1)                             # H/32
                conv7 = conv(conv6, 4096, 1, 1)                             # H/32

            with tf.variable_scope('skips'):
                skip1 = conv1
                skip2 = conv2
                skip3 = conv5

            with tf.variable_scope('decoder'):
                upconv5 = upconv(conv7,  256, 3, 2)                         #H/16
                concat5 = tf.concat([upconv5, skip3], 3)
                iconv5  = conv(concat5,  256, 3, 1)

                upconv4 = upconv(iconv5,  256, 3, 2)                        #H/8
                concat4 = tf.concat([upconv4, skip2], 3)
                iconv4  = conv(concat4,  256, 3, 1)

                upconv3 = upconv(iconv4,  96, 3, 2)                         #H/4
                concat3 = tf.concat([upconv3, skip1], 3)
                iconv3  = conv(concat3,  96, 3, 1)

                upconv2 = upconv(iconv3,  48, 3, 2)                         #H/2
                upconv1 = upconv(upconv2,  16, 3, 2)                        #H/1
                self.feature = upconv1

        self.feat_vars = slim.get_model_variables()
        return self.feature

    def classify(model_range, seg_range, feature_lr, classifier_lr):
        feat_opt = tf.train.AdamOptimizer(feature_lr)
        clas_opt = tf.train.AdamOptimizer(classifier_lr)
        for model in model_range:
            for seg in seg_range:
                with tf.variable_scope('classifier-{}-{}'.format(model, seg)):
                    self.preds[(model, seg)] = slim.conv2d(self.feature, 500, [1, 1])
                    self.clas_vars[(model, seg)] = slim.get_model_variables()[-2:]

                with tf.variable_scope('losses-{}-{}'.format(model, seg)):
                    self.losses[(model, seg)] = self.loss(self.labels, self.preds[(model, seg)])
                    grad = tf.gradients(self.losses[(model, seg)], self.feat_vars + self.clas_vars[(model, seg)])
                    train_op_feat = feat_opt.apply_gradients(zip(grad[:-2], self.feat_vars))
                    train_op_clas = clas_opt.apply_gradients(zip(grad[-2:], self.clas_vars[(model, seg)]))
                    self.train_ops[(model, seg)] = tf.group(train_op_feat, train_op_clas)
        return self.losses, self.train_ops
        

    def _loss(self, labels, logits):
        float_labels = tf.cast(labels, tf.float32)

        epsilon = tf.constant(value=1e-4)
        softmax = tf.nn.softmax(logits) + epsilon
        cross_entropy = -tf.reduce_sum(float_labels * tf.log(softmax), reduction_indices=[-1])
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        
        total_pixels = tf.constant(value=conf.width * conf.height, dtype=tf.float32)
        valid_pixels = tf.reduce_sum(float_labels)
        loss = tf.divide(tf.multiply(cross_entropy_mean, total_pixels), valid_pixels)

        return loss

    def _conv_block(self, x, num_out_layers, kernel_size):
        conv1 = self._conv(x,     num_out_layers, kernel_size, 1)
        conv2 = self._conv(conv1, num_out_layers, kernel_size, 2)
        return conv2

    def _conv(self, x, num_out_layers, kernel_size, stride, activation_fn=tf.nn.elu):
        p = np.floor((kernel_size - 1) / 2).astype(np.int32)
        p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
        return slim.conv2d(p_x, num_out_layers, kernel_size, stride, 'VALID', activation_fn=activation_fn)

    def _deconv(self, x, num_out_layers, kernel_size, scale):
        p_x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]])
        conv = slim.conv2d_transpose(p_x, num_out_layers, kernel_size, scale, 'SAME')
        return conv[:,3:-1,3:-1,:]

    def _upconv(self, x, num_out_layers, kernel_size, scale):
        upsample = self._upsample_nn(x, scale)
        conv = self._conv(upsample, num_out_layers, kernel_size, 1)
        return conv

    def _upsample_nn(self, x, ratio):
        s = tf.shape(x)
        h = s[1]
        w = s[2]
        return tf.image.resize_nearest_neighbor(x, [h * ratio, w * ratio])

    def _maxpool(self, x, kernel_size):
        p = np.floor((kernel_size - 1) / 2).astype(np.int32)
        p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
        return slim.max_pool2d(p_x, kernel_size)
                    