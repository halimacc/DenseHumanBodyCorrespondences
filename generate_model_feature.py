import os
from config import *
import numpy as np
import cv2 as cv
from net import *
from colorutil import *
from meshutil import *

def generate_model_feature(name, swi_range, dis_range, rot_range, sess, input, output, num_vertex):
    print("Generate feature for model {}...".format(model))
    cnt = np.zeros([num_vertex], dtype=np.int32)
    feat = np.zeros([num_vertex, 16], dtype=np.float32)
    for mesh in range(conf.num_meshes[model]):
        print('mesh {}'.format(mesh))
        for swi in swi_range:
            for dis in dis_range:
                for rot in rot_range:
                    view_name = conf.view_name(swi, dis, rot)
                    depth_view_path = conf.depth_view_path(model, mesh, view_name)
                    depth_img = cv.imread(depth_view_path, -1)
                    features = sess.run(output, feed_dict={input: depth_img.reshape([1, 512, 512, 1])}).reshape([512, 512, 16])

                    vertex_view_path = conf.vertex_view_path(model, mesh, view_name)
                    vertex_img = cv.imread(vertex_view_path, -1)
                    vertex = image_color2idx(vertex_img, rgb=False)

                    mask = vertex > 0
                    cnt[vertex[mask] - 1] += 1
                    feat[vertex[mask] - 1, :] += features[mask, :]
    for i in range(num_vertex):
        if cnt[i] > 0:
            feat[i] /= cnt[i]
    feature_path = conf.model_feature_path(model, name)
    create_dirs(feature_path)
    np.save(feature_path, feat)
    return feat

def load_model_color(model_color_path):
    with open(model_color_path, 'r') as file:
        colors = file.read().split('\n')[:-1]
        colors = list(map(lambda s: list(reversed(list(map(lambda x: int(255 * float(x)), s.split(' '))))), colors))
    return np.array(colors)


def load_error_color():
    error_color_path = os.path.join(conf.data_dir, 'error_color.txt')
    with open(error_color_path, 'r') as file:
        colors = file.read().split('\n')[:-1]
        colors = list(map(lambda s: list(reversed(list(map(int, s.split(' '))))), colors))
    return np.array(colors)

if __name__ == '__main__':
    model = 'SCAPE'
    swi_range = [25, 35, 45]
    dis_range = [200, 250]
    rot_range = [i for i in range(0, 360, 15)]
    params = os.path.join('nalex-SM-5', 'model-105000')
    ckpt_path = os.path.join(conf.project_dir, 'log', params)

    with tf.Graph().as_default():
        print("Prepare inference network...")
        input = tf.placeholder(tf.float32, [None, 512, 512, 1])
        net = RHBC('test', input)
        feature_saver = tf.train.Saver(var_list=net.feat_vars)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            feature_saver.restore(sess, ckpt_path)

            print("Prepare reference mesh model...")
            sample_mesh_path = conf.mesh_path(model, 0)
            vertices, faces = load_mesh(sample_mesh_path)
            num_vertex = vertices.shape[0]

            model_feature = generate_model_feature(params, swi_range, dis_range, rot_range, sess, input, net.feature, num_vertex)
