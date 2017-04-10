import numpy as np
import cv2 as cv
import tensorflow as tf
from dhbc import DHBC_FCN

model_path = "D:\\Project\\DHBC\\log\\checkpoints\\model-13000"

if __name__ == '__main__':
    b2f = []
    f2b = []
    for i in range(256):
        b2f.append(i / 255.)
        f2b.append(int(b2f[i] * 255) - i)
    print(b2f)
    print(f2b)
        
        
        
    
    if 0:
        with tf.device('/gpu:0'):
            tf.reset_default_graph()
            
            x_depth = tf.placeholder(tf.float32, [None, 512, 512, 1])
            y_label_dense = tf.placeholder(tf.float32, [None, 512, 512, 500])
        
            model = DHBC_FCN()
            model.build(x_depth)
            
            saver = tf.train.Saver(max_to_keep=10)
            
            with tf.Session() as sess:
                print("Initialize variables...")
                sess.run(tf.global_variables_initializer())
                
                saver.restore(sess, model_path)
            
            