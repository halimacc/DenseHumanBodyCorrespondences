import numpy as np
import cv2 as cv
import tensorflow as tf
from dhbc import DHBC_FCN
import config
import kdtree

class Item(object):
    def __init__(self, key, value):
        self.key = key
        self.value = value

    def __len__(self):
        return len(self.key)

    def __getitem__(self, i):
        return self.key[i]
    
    def __repr__(self):
        return 'Item({}, {})'.format(self.key, self.value)

model_path = "D:\\Project\\DHBC\\log-deconv\\checkpoints\\model-22000"

if __name__ == '__main__':
    src_depth_path = config.get_depth_view_path(0, 0, 0)
    src_depth = cv.imread(src_depth_path, -1)
    
    dst_depth_path = config.get_depth_view_path(0, 0, 42)
    dst_depth = cv.imread(dst_depth_path, -1)
    
    src_label_path = config.get_vertex_view_path(0, 0, 0)
    src_label = cv.imread(src_label_path, -1)
    
    dst_label_path = config.get_vertex_view_path(0, 0, 42)
    dst_label = cv.imread(dst_label_path, -1)
    
    with tf.device('/gpu:0'):
        tf.reset_default_graph()
        
        x_depth = tf.placeholder(tf.float32, [None, 512, 512, 1])
    
        model = DHBC_FCN()
        model.build(x_depth)
        
        variables = []
        saver = tf.train.Saver(max_to_keep=10)
        
        with tf.Session() as sess:
            print("Initialize variables...")
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, model_path)
        
            src_feat = sess.run([model.deconv11], feed_dict={x_depth: src_depth.reshape([1, 512, 512, 1])})[0].reshape(512, 512, 16)
            dst_feat = sess.run([model.deconv11], feed_dict={x_depth: dst_depth.reshape([1, 512, 512, 1])})[0].reshape(512, 512, 16)
            
            cands = [Item(src_feat[i, j], (i, j)) for i in range(512) for j in range(512) if src_depth[i, j] > 0]
            print(cands[0])
            kdt = kdtree.create(cands)
            
            a, b = 0, 0
            for i in range(512):
                print("tick" + str(i))
                for j in range(512):
                    if dst_depth[i, j] > 0:
                        b += 1
                        #print(kdt.search_nn(dst_feat[i, j]))
                        ii, jj = kdt.search_nn(dst_feat[i, j])[0].data.value
                        if src_label[ii, jj, 0] == dst_label[i, j, 0] and src_label[ii, jj, 1] == dst_label[i, j, 1] and src_label[ii, jj, 2] == dst_label[i, j, 2]: 
                            a += 1
            print(a)
            print(b)
            print(a / b)
                        
                        
            
            