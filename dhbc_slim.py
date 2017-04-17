import tensorflow as tf
import tensorflow.contrib.slim as slim
import config

class DHBC_FCN(object):    
    def inference(self, depth, padding='SAME', train=False):
        self.conv1 = slim.conv2d(depth, 96, [11, 11], 4, padding, scope='conv1')
        self.pool1 = slim.max_pool2d(self.conv1, [3, 3], 2, padding, scope='pool1')
        self.conv2 = slim.conv2d(self.pool1, 256, [5, 5], 1, padding, scope='conv2')
        self.pool2 = slim.max_pool2d(self.conv2, [3, 3], 2, padding, scope='pool2')
        self.conv3 = slim.conv2d(self.pool2, 384, [3, 3], 1, padding, scope='conv3')
        self.conv4 = slim.conv2d(self.conv3, 384, [3, 3], 1, padding, scope='conv4')
        self.conv5 = slim.conv2d(self.conv4, 256, [3, 3], 1, padding, scope='conv5')
        self.pool5 = slim.max_pool2d(self.conv5, [3, 3], 2, padding, scope='pool5')
        self.fc6 = slim.conv2d(self.pool5, 4096, [1, 1], 1, padding, scope='fc6')
        if train:
            self.fc6 = slim.dropout(self.fc6)
        self.fc7 = slim.conv2d(self.fc6, 4096, [1, 1], 1, padding, scope='fc7')
        if train:
            self.fc7 = slim.dropout(self.fc7)
        self.deconv8 = slim.conv2d_transpose(self.fc7, 384, [3, 3], 2, padding, scope='deconv8')
        self.deconv9 = slim.conv2d_transpose(self.deconv8, 256, [3, 3], 2, padding, scope='deconv9')
        self.deconv10 = slim.conv2d_transpose(self.deconv9, 96, [5, 5], 2, padding, scope='deconv10')
        self.deconv11 = slim.conv2d_transpose(self.deconv10, 16, [11, 11], 4, padding, scope='deconv11')
        self.pred = slim.conv2d(self.deconv11, 500, [1, 1], 1, padding, scope='pred')
        
    def get_branch_name(self, model_idx, segmentation_idx):
        return 'model_{}_seg_{}'.format(str(model_idx).zfill(2), str(segmentation_idx).zfill(3))
        
    def inference_sas(self, depth):
        padding = 'VALID'
        self.conv1 = slim.conv2d(depth, 96, [11, 11], 1, padding, scope='conv1')
        self.conv2 = slim.conv2d(self.conv1, 256, [5, 5], 1, padding, scope='conv2')
        self.conv3 = slim.conv2d(self.conv2, 384, [3, 3], 1, padding, scope='conv3')
        self.conv4 = slim.conv2d(self.conv3, 384, [3, 3], 1, padding, scope='conv4')
        self.conv5 = slim.conv2d(self.conv4, 256, [3, 3], 1, padding, scope='conv5')
        self.fc6 = slim.conv2d(self.conv5, 4096, [1, 1], 1, padding, scope='fc6')
        self.fc7 = slim.conv2d(self.fc6, 4096, [1, 1], 1, padding, scope='fc7')
        self.upsample = slim.conv2d(self.fc7, 16, [1, 1], 1, padding, scope='upsample')
        self.preds = {}
        for model_idx in range(1):
            for segmentation_idx in range(config.num_seg):
                branch= self.get_branch_name(model_idx, segmentation_idx)
                self.preds[branch] = slim.conv2d(self.upsample, 500, [1, 1], 1, padding, scope=branch)