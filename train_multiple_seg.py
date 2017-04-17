import tensorflow as tf
import numpy as np
#from dhbc import DHBC_FCN
from dhbc_slim import DHBC_FCN
from losses import softmax_cross_entropy
import config
import random
import cv2 as cv
import os
import matplotlib.cm as cm
import time
import tensorflow.contrib.slim as slim

project_path = 'D:\\Project\\DHBC\\'
start_step = 2000

# train
learning_rate = 0.0001
training_iters = 200000
batch_size = 1

# log and display
logs_dir = project_path + 'log\\'
checkpoints_dir = logs_dir + 'checkpoints\\'
checkpoint_step = 1000

display_dir = logs_dir + 'display\\'
display_step = 20

random.seed()

# prepare color transition
color2int_dense = {(0, 0, 0): 0}
int2color_dense = {0: (0, 0, 0)}
for i in range(config.num_dense_color):
    color = config.int2color(i + 1, config.num_dense_color + 1)
    color = (int(255 * color[0]), int(255 * color[1]), int(255 * color[2]))
    int2color_dense[i + 1] = color
    color2int_dense[color] = i + 1
    
def random_batch(batch_size):
    model_idx = 0
    segment_idx = 0
    mesh_idx = random.randint(0, config.num_mesh[model_idx] - 1)

    view_indices = random.sample([i for i in range(144)], batch_size)
    batch_depth = np.zeros([batch_size, 512, 512, 1], dtype=np.uint8)
    batch_label_dense = np.zeros([batch_size, 512, 512, 500], dtype=np.uint8)
    batch = 0
    for view_idx in view_indices:
        depth_path = config.get_depth_view_path(model_idx, mesh_idx, view_idx)
        depth_img = cv.imread(depth_path, 0)
        batch_depth[batch,:,:,:] = depth_img.reshape([512, 512, 1])
        
        color_path = config.get_segmentation_view_path(model_idx, mesh_idx, segment_idx, view_idx) 
        color_img = cv.imread(color_path, -1)
        for i in range(512):
            for j in range(512):
                color = (color_img[i, j, 0], color_img[i, j, 1], color_img[i, j, 2])
                idx = color2int_dense[color]
                if idx > 0:
                    batch_label_dense[batch, i, j, idx - 1] = 1
                    
        batch += 1
    return batch_depth, batch_label_dense

def display(step, prediction, label):
    prediction, label = prediction[0], label[0]
    #mul = np.multiply(prediction, label)
    labelimg = np.zeros([512, 512, 3], dtype=np.uint8)
    predimg = np.zeros([512, 512, 3], dtype=np.uint8)
    #heatmap = np.zeros([512, 512, 3], dtype=np.uint8)
    correct = np.zeros([512, 512, 1], dtype=np.uint8)
    for i in range(512):
        for j in range(512):
            if np.sum(label[i, j]) > 0:
                idx_label = np.argmax(label[i, j])
                idx_pred = np.argmax(prediction[i, j])
                color = int2color_dense[idx_label + 1]
                for c in range(3):
                    labelimg[i, j, c] = color[c]
                color = int2color_dense[idx_pred + 1]
                for c in range(3):
                    predimg[i, j, c] = color[c]
#                color = cm.hot(mul[i, j, idx_pred])
#                for c in range(3):
#                    heatmap[i, j, c] = int(color[c] * 256)
                correct[i, j, 0] = 255 if idx_label == idx_pred else 0
    display_path = display_dir + str(step).zfill(8)
    cv.imwrite(display_path + '_groundtruth.png', labelimg)
    cv.imwrite(display_path + '_prediction.png', predimg)
    #cv.imwrite(display_path + '_heatmap.png', heatmap)
    cv.imwrite(display_path + '_correctness.png', correct)
    
def timeit(t=None):
    if t:
        print("Time elapse: " + str(time.time() - t))
    return time.time()

if __name__ == '__main__':
    
    with tf.device('/gpu:0'):
        tf.reset_default_graph()
        
        x_depth = tf.placeholder(tf.float32, [None, 512, 512, 1])
        y_label_dense = tf.placeholder(tf.float32, [None, 512, 512, 500])
    
        t = timeit()
        print("Set up inference model...")
        
        model = DHBC_FCN()
        model.inference(x_depth, train=True)
        
        t = timeit(t)
        print("Set up cost and optimizer...")
        
        softmax_cross_entropy(model.pred, y_label_dense)
        total_loss = tf.losses.get_total_loss()
        optimizer = tf.train.AdamOptimizer().minimize(total_loss)
        
        saver = tf.train.Saver(max_to_keep=10)
        
        with tf.Session() as sess:
            t = timeit(t)
            print("Initialize variables...")
            sess.run(tf.global_variables_initializer())
            
            ckpt_path = checkpoints_dir + 'model-' + str(start_step)
            if os.path.exists(ckpt_path + '.meta'):
                t = timeit(t)
                print("Restore model at step {0}...".format(start_step))
                saver.restore(sess, ckpt_path)
                step = start_step + 1
            else:
                print("No snapshot at step {0}, training from scratch...".format(start_step))
                step = 1
                
            t = timeit(t)
            pred_name = config.get_pred_name(0, 0)
            while step * batch_size < training_iters:
                batch_depth, batch_label_dense = random_batch(batch_size)
                _, loss= sess.run([optimizer, total_loss], feed_dict={x_depth: batch_depth, y_label_dense: batch_label_dense})
                print("Step " + str(step * batch_size) + ", Loss= {:.6f}".format(loss))
                
                if step % display_step == 0:
                    pred = sess.run(tf.nn.softmax(model.pred), feed_dict={x_depth: batch_depth, y_label_dense: batch_label_dense})
                    display(step, pred, batch_label_dense)
                    
                if step % checkpoint_step == 0:
                    saver.save(sess, checkpoints_dir + 'model', step)
                    print("Save model at step {0}.".format(step))
                step += 1
            print("Optimization Finished!")