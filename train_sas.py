import tensorflow as tf
import numpy as np
from dhbc_slim import DHBC_FCN
from losses import softmax_cross_entropy
import config
import random
import cv2 as cv
import os
import time
import tensorflow.contrib.slim as slim

project_path = 'D:\\Project\\DHBC\\'
start_step = 0

# train
learning_rate = 0.0001
training_iters = 200000
batch_size = 128

# log and display
logs_dir = project_path + 'log-sas\\'
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
    
def random_batch(model_idx, segmentation_idx, batch_size):
    batch_depth = np.zeros([batch_size, 21, 21, 1], dtype=np.uint8)
    batch_label_dense = np.zeros([batch_size, 1, 1], dtype=np.int64)
    
    view_cands = random.sample([(i, j) for i in range(config.num_mesh[model_idx]) for j in range(144)], batch_size)
    batch = 0
    for view in view_cands:
        depth_view_path = config.get_depth_view_path(model_idx, view[0], view[1])
        depth_view = cv.imread(depth_view_path, -1)
        while 1:
            x, y = random.randint(0, 512 - 21), random.randint(0, 512 - 21)
            if depth_view[x + 10, y + 10] > 0:
                break
        batch_depth[batch, :, :, 0] = depth_view[x:x+21, y:y+21]
        
        segmentation_view_path = config.get_segmentation_view_path(model_idx, view[0], segmentation_idx, view[1])
        segmentation_view = cv.imread(segmentation_view_path, -1)
        x, y = x + 10, y + 10
        color = (segmentation_view[x, y, 0], segmentation_view[x, y, 1], segmentation_view[x, y, 2])
        batch_label_dense[batch, 0, 0] = color2int_dense[color] - 1
        batch += 1
    return batch_depth, batch_label_dense
    
def timeit(t=None):
    if t:
        print("Time elapse: " + str(time.time() - t))
    return time.time()

if __name__ == '__main__':
    
    with tf.device('/gpu:0'):
        tf.reset_default_graph()
        
        x_depth = tf.placeholder(tf.float32, [None, 21, 21, 1])
        y_label_dense = tf.placeholder(tf.int64, [None, 1, 1])
    
        t = timeit()
        print("Set up inference model...")
        
        model = DHBC_FCN()
        model.inference_sas(x_depth)
        
        t = timeit(t)
        print("Set up cost and optimizer...")
        
        costs = {}
        optimizers = {}
        accuracies = {}
        for model_idx in range(1):
            for segmentation_idx in range(config.num_seg):
                branch = model.get_branch_name(model_idx, segmentation_idx)
                costs[branch] = tf.losses.sparse_softmax_cross_entropy(labels=y_label_dense, logits=model.preds[branch], loss_collection=None)
                optimizers[branch] = tf.train.AdamOptimizer().minimize(costs[branch])
                accuracies[branch] = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(model.preds[branch], axis=-1), y_label_dense), tf.float32))
        
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
            while step * batch_size < training_iters:
                model_idx = random.randint(0, 0)
                segmentation_idx = random.randint(0, config.num_seg - 1)
                branch = model.get_branch_name(model_idx, segmentation_idx)                
                batch_depth, batch_label_dense = random_batch(model_idx, segmentation_idx, batch_size)
                _, loss, acc= sess.run([optimizers[branch], costs[branch], accuracies[branch]], feed_dict={x_depth: batch_depth, y_label_dense: batch_label_dense})
                print(branch + " Step " + str(step) + ", Loss= {:.6f}".format(loss) + ", Accuracy= {:.6f}%".format(acc * 100))
                
                #if step % display_step == 0:
                    #pred = sess.run(tf.nn.softmax(model.pred), feed_dict={x_depth: batch_depth, y_label_dense: batch_label_dense})
                    #display(step, pred, batch_label_dense)
                    
                if step % checkpoint_step == 0:
                    saver.save(sess, checkpoints_dir + 'model', step)
                    print("Save model at step {0}.".format(step))
                step += 1
            print("Optimization Finished!")