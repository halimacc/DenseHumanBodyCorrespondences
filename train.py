import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'
os.environ['CUDA_VISIBILE_DEVICES']='0'

from config import *
import cv2 as cv
import numpy as np
from net import *
from colorutil import *

def decode_label(seg_view):
    label = np.zeros([conf.height, conf.width, conf.num_classes], np.uint8)
    seg = image_color2idx(seg_view, rgb=True)
    mask = seg > 0
    label[mask, seg[mask] - 1] = 1
    return label

def random_batch(batch_size):
    input_queue = tf.train.string_input_producer([conf.tmp_file_path])
    line_reader = tf.TextLineReader()
    _, line = line_reader.read(input_queue)
    data_paths = tf.string_split([line]).values

    depth = tf.image.decode_png(tf.read_file(data_paths[0]), 0)

    seg = tf.image.decode_png(tf.read_file(data_paths[1]), 0)
    label = tf.py_func(decode_label, [seg], [tf.uint8])[0]

    depth.set_shape([conf.height, conf.width, 1])
    label.set_shape([conf.height, conf.width, conf.num_classes])

    depth = tf.cast(depth, tf.float32)

    depth_batch, label_batch = tf.train.batch([depth, label], batch_size, num_threads=8)
    return depth_batch, label_batch

def train(model_range, seg_range, swi_range, dis_range, rot_range, batch_size=1, num_epochs=10, learning_rate=1e-4, log_dir='', checkpoint_path='', retrain=False):
    steps_per_ckpt = 5000
    #loss = np.zeros(steps_per_ckpt, np.float32)
    #acc = np.zeros(steps_per_ckpt, np.float32)

    # preprocess
    print("Prepare random file list...")
    all_task_list = []
    for i in range(num_epochs):
        task_list = []
        for model in model_range:
            for mesh in range(conf.num_meshes[model]):
                for seg in seg_range:
                    for swi in swi_range:
                        for dis in dis_range:
                            for rot in rot_range:
                                view_name = conf.view_name(swi, dis, rot)
                                depth_view_path = conf.depth_view_path(model, mesh, view_name)
                                seg_view_path = conf.segmentation_view_path(model, mesh, seg, view_name)
                                file_paths = '{} {}\n'.format(depth_view_path, seg_view_path)
                                task_list.append(((model, seg), file_paths))
        #np.random.shuffle(task_list)
        all_task_list += task_list
    np.random.shuffle(all_task_list)

    with open(conf.tmp_file_path, 'w') as file:
        for task in all_task_list:
            file.write(task[1])

    with tf.Graph().as_default():
        global_step = tf.Variable(0, trainable=False)

        num_training_samples = len(seg_range) * len(swi_range) * len(dis_range) * len(rot_range) * np.sum([conf.num_meshes[model] for model in model_range])
        print("Training parameters:")
        print('\tModels: {}'.format(model_range))
        print('\tSegmentation range: {}'.format(seg_range))
        print('\tTotal training samples: {}'.format(num_training_samples))
        steps_per_epoch = np.ceil(num_training_samples / batch_size).astype(np.int32)
        num_total_steps = num_epochs * steps_per_epoch
        print("\tTotal number of steps: {}".format(num_total_steps))

        # start_learning_rate = learning_rate
        # boundaries = [np.int32((3/5) * num_total_steps), np.int32((4/5) * num_total_steps)]
        # values = [learning_rate, learning_rate / 2, learning_rate / 4]
        # learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)

        learning_rate_factor = len(set(model_range)) * len(set(seg_range))
        print("\tTraining tasks: {}".format(learning_rate_factor))

        feat_opt = tf.train.AdamOptimizer(learning_rate / learning_rate_factor)
        clas_opt = tf.train.AdamOptimizer(learning_rate)

        depth_batch, label_batch = random_batch(batch_size)

        #with tf.device('/gpu:0'):
        net = RHBC('train', depth_batch, label_batch, feat_opt, clas_opt)

        config = tf.ConfigProto(allow_soft_placement=True)
        with tf.Session(config=config) as sess:

            # total_num_parameters = 0
            # for variable in tf.trainable_variables():
            #     total_num_parameters += np.array(variable.get_shape().as_list()).prod()
            # print("\tNumber of trainable parameters: {}".format(total_num_parameters))

            print("Initial network...")
            #train_saver = tf.train.Saver(var_list=net.feat_vars + net.clas_vars[('SCAPE', 0)])
            train_saver = tf.train.Saver(max_to_keep=None)

            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            coordinator = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

            if checkpoint_path != '':
                print("Load checkpoint from {}".format(checkpoint_path))
                train_saver.restore(sess, checkpoint_path)
                if retrain:
                    sess.run(global_step.assign(0))

            
            start_step = global_step.eval(session=sess)
            start_time = time.time()

            print("Start training...")
            vis_time = time.time()
            step_per_vis = 100
            loss_per_vis = 0
            for step in range(start_step, len(all_task_list)):
                task = all_task_list[step][0]

                if step and step % step_per_vis == 0:
                    _, loss_value, acc_value, gt, pred = sess.run([net.train_ops[task], net.losses[task], net.accuracy[task], label_batch, net.preds[task]])
                    loss_per_vis += loss_value

                    print_string = 'step {:>6} | examples/s: {:4.2f} | loss: {:.5f} | acc: {:.5f} | time elapsed: {:.2f}h | time left: {:.2f}h'
                    duration = time.time() - vis_time
                    examples_per_sec =  step_per_vis / duration
                    time_sofar = (time.time() - start_time) / 3600
                    training_time_left = (num_total_steps - step) / examples_per_sec / 3600
                    print(print_string.format(step, examples_per_sec, loss_per_vis / step_per_vis, acc_value, time_sofar, training_time_left))

                    mask = np.sum(gt[0], -1).astype(np.uint8)
                    class_gt = np.argmax(gt[0], -1)
                    class_pred = np.argmax(pred[0], -1)
                    acc = np.expand_dims(255 * mask * (class_gt == class_pred).astype(np.uint8), -1)
                    acc_path = os.path.join(log_dir, 'vis', 'acc-{:06d}.jpg'.format(step))
                    cv.imwrite(acc_path, acc)

                    vis_time = time.time()
                    loss_per_vis = 0
                else:
                    _, loss_value = sess.run([net.train_ops[task], net.losses[task]])
                    loss_per_vis += loss_value
                
                if step and step % steps_per_ckpt == 0:
                    train_saver.save(sess, os.path.join(log_dir, 'model'), global_step=step)
                    #np.save(os.path.join(log_dir, 'loss-{}.npy'.format(step)), loss)

            train_saver.save(sess, os.path.join(log_dir, 'model'), global_step=num_total_steps)
    
            coordinator.request_stop()
            coordinator.join(threads)

if __name__ == '__main__':
    #model_name = 'alex-skip-1-5-2'
    #checkpoint_path = os.path.join(conf.project_dir, 'log', 'alex-skip-1-5', 'model-85200')
    model_name = 'nalex-SM-5'
    create_dirs(os.path.join(conf.project_dir, 'log', model_name, 'vis', 'test'))
    checkpoint_path = os.path.join(conf.project_dir, 'log', 'nalex-SM-1', 'model-49392')
    #checkpoint_path = ''

    np.random.seed()
    model_range = ['SCAPE', 'MITcrane', 'MITjumping', 'MITmarch1', 'MITsquat1', 'MITsamba', 'MITswing']
    #seg_range = [i for i in range(5)] + [1, 2, 3, 4]
    seg_range = [0, 1, 2, 3, 4]
    swi_range = [25, 35, 45]
    dis_range = [200, 250]
    rot_range = [i for i in range(0, 360, 15)]
    batch_size = 1
    learning_rate = 1e-4
    log_dir = os.path.join(conf.project_dir, 'log', model_name)
    retrain = False
    train(model_range, seg_range, swi_range, dis_range, rot_range,
        batch_size=batch_size,
        num_epochs=1,
        learning_rate=learning_rate,
        log_dir=log_dir,
        checkpoint_path=checkpoint_path,
        retrain=retrain
        )
