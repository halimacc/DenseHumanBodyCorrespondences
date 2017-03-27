import tensorflow as tf
import numpy as np
from model import FCNDHBC
from loss import loss
import config
import random
import cv2 as cv
import time
import os
import matplotlib.cm as cm

start_step = 20
snapshot_step = 5000
snapshot_path = 'D:\\Project\\DHBC\\model\\{0}\\'

learning_rate = 0.001
training_iters = 200000
batch_size = 1

display_step = 1
visualize_step = 1
visualize_path = 'D:\\Project\\DHBC\\tmp\\{0}.png'

logs_path = "D:\\Project\\DHBC\\model\\log1"

random.seed()

def random_batch(batch_size):
    model_idx = 0
    segment_idx = 0
    mesh_idx = random.randint(0, config.num_mesh[model_idx] - 2)

    # special case for model 0
    if model_idx == 0 and mesh_idx >= 51:
        mesh_idx += 1

    view_indices = random.sample([i for i in range(144)], batch_size)
    batch_depth = np.zeros([batch_size, 512, 512, 1], dtype=np.uint8)
    batch_label_500 = np.zeros([batch_size, 512, 512, 500], dtype=np.uint8)
    batch = 0
    for view_idx in view_indices:
        color_path, depth_path = config.get_view_path(model_idx, segment_idx, mesh_idx, view_idx)
        
        depth_img = cv.imread(depth_path, 0)
        batch_depth[batch,:,:,:] = depth_img.reshape([512, 512, 1])
        color_img = cv.imread(color_path, -1)
        for i in range(512):
            for j in range(512):
                idx = config.color2int(color_img[i, j])
                if idx > 0:
                    batch_label_500[batch, i, j, idx - 1] = 1
        batch += 1
    return batch_depth, batch_label_500

def visualize(step, prediction, label):
    mul = np.multiply(prediction, label)
    heatmap = np.zeros([512, 512, 3], dtype=np.uint8)
    for i in range(512):
        for j in range(512):
            if np.sum(label[i, j]) > 0:
                idx = np.argmax(label[i, j])
                color = cm.hot(mul[i, j, idx])
                for c in range(3):
                    heatmap[i, j, 2 - c] = 255 if color[c] == 1 else int(color[c] * 256)
    cv.imwrite(visualize_path.format(step), heatmap)

if __name__ == '__main__':
    
    with tf.device('/gpu:0'):
        tf.reset_default_graph()
        
        depth = tf.placeholder(tf.float32, [None, 512, 512, 1])
        
        label_500 = tf.placeholder(tf.float32, [None, 512, 512, 500])
    
        model = FCNDHBC()
        model.build(depth)
    
        cost = tf.reduce_mean(loss(logits=model.pred, labels=label_500, num_classes=500))
        #cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model.pred, labels=label_500 ))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    
        # Initializing the variables
        init = tf.global_variables_initializer()
        
        tf.summary.scalar("training_loss", cost)
        
        saver = tf.train.Saver()
    
        # Launch the graph
        with tf.Session() as sess:
            sess.run(init)
            
            if os.path.exists(snapshot_path.format(start_step)):
                saver.restore(sess, snapshot_path.format(start_step) + "snapshot.ckpt")
                print("Load model at step {0}.".format(start_step))
                step = start_step + 1
            else:
                step = 1
                
            summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
            merged_summary_op = tf.summary.merge_all()
                
            # Keep training until reach max iterations
            while step * batch_size < training_iters:
                batch_depth, batch_label_500 = random_batch(batch_size)
                # Run optimization op (backprop)
                _, loss, summary = sess.run([optimizer, cost, merged_summary_op], feed_dict={depth: batch_depth, label_500: batch_label_500})
                summary_writer.add_summary(summary, step)
                print("Iter " + str(step * batch_size) + ", Loss= {:.6f}".format(loss))
                
                if step % display_step == 0:
                    pred = sess.run(tf.nn.softmax(model.pred), feed_dict={depth: batch_depth, label_500: batch_label_500})
                    visualize(step, pred[0], batch_label_500[0])
                    
                if step % snapshot_step == 0:
                    os.mkdir(snapshot_path.format(step))
                    save_path = snapshot_path.format(step) + "snapshot.ckpt"
                    saver.save(sess, save_path)
                    print("Save model at step {0}.".format(step))
                step += 1
            print("Optimization Finished!")