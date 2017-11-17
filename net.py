from config import *
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np


class RHBC:
    def __init__(self, mode, depth, labels=None, feat_opt=None, clas_opt=None):
        self.mode = mode
        self.depth = depth
        self.labels = labels
        self.feat_opt = feat_opt
        self.clas_opt = clas_opt

        #self.build_feature(self.mode)
        #self.build_vgg()
        self.build_alex()

        if self.mode == 'test':
            return 

        self.build_classifier()

    def build_alex(self):
        conv = self.conv
        upconv = self.upconv
        maxpool = self.maxpool
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

                # with tf.variable_scope('decoder'):
                #     deconv8 = slim.conv2d_transpose(fc7, 256, [3, 3], 2)        # H/16
                #     deconv9 = slim.conv2d_transpose(deconv8, 256, [3, 3], 2)    # H/8
                #     deconv10 = slim.conv2d_transpose(deconv9, 96, [5, 5], 2)    # H/4
                #     deconv11 = slim.conv2d_transpose(deconv10, 16, [11, 11], 4)
                #     self.feature = deconv11

        self.feat_vars = slim.get_model_variables()

    def build_vgg(self):
        #set convenience functions
        conv = self.conv
        upconv = self.upconv

        with tf.variable_scope('encoder'):
            conv1 = self.conv_block(self.depth,  32, 7) # H/2
            conv2 = self.conv_block(conv1,       64, 5) # H/4
            conv3 = self.conv_block(conv2,      128, 3) # H/8
            conv4 = self.conv_block(conv3,      256, 3) # H/16
            conv5 = self.conv_block(conv4,      512, 3) # H/32
            conv6 = self.conv_block(conv5,      512, 3) # H/64
            conv7 = self.conv_block(conv6,      512, 3) # H/128

        with tf.variable_scope('skips'):
            skip1 = conv1
            skip2 = conv2
            skip3 = conv3
            skip4 = conv4
            skip5 = conv5
            skip6 = conv6
        
        with tf.variable_scope('decoder'):
            upconv7 = upconv(conv7,  512, 3, 2) #H/64
            concat7 = tf.concat([upconv7, skip6], 3)
            iconv7  = conv(concat7,  512, 3, 1)

            upconv6 = upconv(iconv7, 512, 3, 2) #H/32
            concat6 = tf.concat([upconv6, skip5], 3)
            iconv6  = conv(concat6,  512, 3, 1)

            upconv5 = upconv(iconv6, 256, 3, 2) #H/16
            concat5 = tf.concat([upconv5, skip4], 3)
            iconv5  = conv(concat5,  256, 3, 1)

            upconv4 = upconv(iconv5, 128, 3, 2) #H/8
            concat4 = tf.concat([upconv4, skip3], 3)
            iconv4  = conv(concat4,  128, 3, 1)
            #self.disp4 = self.get_disp(iconv4)
            #udisp4  = self.upsample_nn(self.disp4, 2)

            upconv3 = upconv(iconv4,  64, 3, 2) #H/4
            #concat3 = tf.concat([upconv3, skip2, udisp4], 3)
            concat3 = tf.concat([upconv3, skip2], 3)
            iconv3  = conv(concat3,   64, 3, 1)
            #self.disp3 = self.get_disp(iconv3)
            #udisp3  = self.upsample_nn(self.disp3, 2)

            upconv2 = upconv(iconv3,  32, 3, 2) #H/2
            #concat2 = tf.concat([upconv2, skip1, udisp3], 3)
            concat2 = tf.concat([upconv2, skip1], 3)
            iconv2  = conv(concat2,   32, 3, 1)
            #self.disp2 = self.get_disp(iconv2)
            #udisp2  = self.upsample_nn(self.disp2, 2)

            upconv1 = upconv(iconv2,  16, 3, 2) #H
            #concat1 = tf.concat([upconv1, udisp2], 3)
            #iconv1  = conv(concat1,   16, 3, 1)
            #self.disp1 = self.get_disp(iconv1)
            self.feature = upconv1

        self.feat_vars = slim.get_model_variables()

    def build_classifier(self):
        self.preds = {}
        self.clas_vars = {}
        self.losses = {}
        self.accuracy = {}
        self.train_ops = {}
        for model in conf.train_model_range:
            for seg in conf.train_seg_range:
                with tf.variable_scope('classifier-{}-{}'.format(model, seg)):
                    self.preds[(model, seg)] = slim.conv2d(self.feature, 500, [1, 1])
                    self.clas_vars[(model, seg)] = slim.get_model_variables()[-2:]

                with tf.variable_scope('losses-{}-{}'.format(model, seg)):
                    self.losses[(model, seg)], self.accuracy[(model, seg)] = self.loss_and_acc(self.labels, self.preds[(model, seg)])
                    grad = tf.gradients(self.losses[(model, seg)], self.feat_vars + self.clas_vars[(model, seg)])
                    train_op_feat = self.feat_opt.apply_gradients(zip(grad[:-2], self.feat_vars))
                    train_op_clas = self.clas_opt.apply_gradients(zip(grad[-2:], self.clas_vars[(model, seg)]))
                    self.train_ops[(model, seg)] = tf.group(train_op_feat, train_op_clas)

    def loss_and_acc(self, labels, logits):
        float_labels = tf.cast(labels, tf.float32)

        epsilon = tf.constant(value=1e-4)
        softmax = tf.nn.softmax(logits) + epsilon
        cross_entropy = -tf.reduce_sum(float_labels * tf.log(softmax), reduction_indices=[-1])
        cross_entropy_mean = tf.reduce_mean(cross_entropy)

        mask = tf.reduce_sum(float_labels, -1)
        predict = tf.argmax(softmax, -1)
        groundtruth = tf.argmax(labels, -1)
        correct_pixels = tf.reduce_sum(tf.multiply(mask, tf.cast(tf.equal(predict, groundtruth), tf.float32)))
        
        total_pixels = tf.constant(value=conf.width * conf.height, dtype=tf.float32)
        valid_pixels = tf.reduce_sum(float_labels)
        loss = tf.divide(tf.multiply(cross_entropy_mean, total_pixels), valid_pixels)
        acc = tf.divide(correct_pixels, valid_pixels)

        return loss, acc

    def conv_block(self, x, num_out_layers, kernel_size):
        conv1 = self.conv(x,     num_out_layers, kernel_size, 1)
        conv2 = self.conv(conv1, num_out_layers, kernel_size, 2)
        return conv2

    def conv(self, x, num_out_layers, kernel_size, stride, activation_fn=tf.nn.elu):
        p = np.floor((kernel_size - 1) / 2).astype(np.int32)
        p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
        return slim.conv2d(p_x, num_out_layers, kernel_size, stride, 'VALID', activation_fn=activation_fn)

    def deconv(self, x, num_out_layers, kernel_size, scale):
        p_x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]])
        conv = slim.conv2d_transpose(p_x, num_out_layers, kernel_size, scale, 'SAME')
        return conv[:,3:-1,3:-1,:]

    def upconv(self, x, num_out_layers, kernel_size, scale):
        upsample = self.upsample_nn(x, scale)
        conv = self.conv(upsample, num_out_layers, kernel_size, 1)
        return conv

    def upsample_nn(self, x, ratio):
        s = tf.shape(x)
        h = s[1]
        w = s[2]
        return tf.image.resize_nearest_neighbor(x, [h * ratio, w * ratio])

    def maxpool(self, x, kernel_size):
        p = np.floor((kernel_size - 1) / 2).astype(np.int32)
        p_x = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]])
        return slim.max_pool2d(p_x, kernel_size)
                    