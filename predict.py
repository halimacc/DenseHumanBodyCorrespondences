import os
from config import *
import numpy as np
import cv2 as cv
from net import *
from colorutil import *
from meshutil import *

def timeit(t=None):
    if t:
        print("Time elapse: " + str(time.time() - t))
    return time.time()


def generate_test_mesh_feature(mesh_idx, sess, input, output, num_vertex):
    print("Generate feature for test mesh {}...".format(mesh_idx))
    cnt = np.zeros([num_vertex], dtype=np.int32)
    feat = np.zeros([num_vertex, 16], dtype=np.float32)
    for view_idx in range(144):
        depth_view_path = config.get_test_depth_view_path(mesh_idx, view_idx)
        depth_img = cv.imread(depth_view_path, -1)

        features = sess.run(output, feed_dict={input: depth_img.reshape([1, 512, 512, 1])}).reshape([512, 512, 16])

        vertex_view_path = config.get_test_vertex_view_path(mesh_idx, view_idx)
        vertex_view = (cv.imread(vertex_view_path, -1) - 2147483648).astype(np.int32)

        accumulate_feature(feat, cnt, features, vertex_view)

    for i in range(num_vertex):
        if cnt[i] > 0:
            feat[i] /= cnt[i]
    return feat


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

def sqrdis(src, dst):
    dists = np.multiply(np.dot(dst, src.T), -2)
    dists += np.sum(np.square(dst), 1, keepdims=True)
    dists += np.sum(np.square(src), 1)
    return dists


def correspond(model_feature, depth_feature, mask):
    feature = depth_feature[mask]

    dists = sqrdis(model_feature, feature)
    return np.argmin(dists, -1)


def correspond2(key_features, clas_features, clas_vertices, depth_features, mask):
    features = depth_features[mask]

    key_dists = sqrdis(key_features, features)
    N = 4
    nearest = np.argpartition(key_dists, N, -1)[:, :N]

    all_match = np.zeros([features.shape[0], N], np.int64)
    all_dists = np.ones([features.shape[0], N]) * 1e10
    for i in range(key_features.shape[0]):
        for j in range(N):
            clas_mask = nearest[:, j] == i
            if clas_mask.shape[0] > 0:
                dst_features = features[clas_mask]
                dists = sqrdis(clas_features[i], dst_features)
                min_dist = np.amin(dists, -1)
                match = np.argmin(dists, -1)
                all_dists[clas_mask, j] = min_dist
                all_match[clas_mask, j] = clas_vertices[i][match]
    min_dists = np.argmin(all_dists, -1)
    return all_match[np.arange(all_match.shape[0]), min_dists]
    return all_match[np.argmin(all_dists, -1)]

    # return all_match


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

def smooth(feature, mask):
    nf = np.zeros_like(feature)
    kernel_cul = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.uint8)
    cul = cv.erode(mask.astype(np.uint8), kernel_cul, iterations=1, borderValue=0)
    cul_cmask = cul.astype(np.bool)
    for i in range(512):
        for j in range(512):
            if cul_cmask[i, j]:
                for x in range(-1, 2):
                    for y in range(-1, 2):
                        nf[i,j] += feature[i + x, y + j]
                nf[i, j] /= 9
    return nf



test = False
model = 'SCAPE'
generate_feature = False
predict_model = True
predict_kinect = True

if __name__ == '__main__':
    swi_range = [25, 35, 45]
    dis_range = [200, 250]
    rot_range = [i for i in range(0, 360, 15)]

    err_colors = load_error_color()
    params = os.path.join('nalex-SM-5', 'model-105000')
    #model_dir = os.path.join('.', 'model', '9-25')
    ckpt_path = os.path.join(conf.project_dir, 'log', params)
    with tf.device('/gpu:0'):
        with tf.Graph().as_default():
            print("Prepare inference network...")
            input = tf.placeholder(tf.float32, [None, 512, 512, 1])
            net = RHBC('test', input)
            feature_saver = tf.train.Saver(var_list=net.feat_vars)
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                feature_saver.restore(sess, ckpt_path)

                if test:
                    for mesh_idx in range(200):
                        test_mesh_path = config.get_test_mesh_path(mesh_idx)
                        vertices, faces = load_mesh(test_mesh_path)
                        features = generate_test_mesh_feature(mesh_idx, swi_range, dis_range, rot_range, sess, input, net.feature, num_vertex)

                        test_feature_path = config.get_test_feature_path(mesh_idx)
                        np.save(test_feature_path, features)
                else:
                    print("Prepare reference mesh model...")
                    sample_mesh_path = conf.mesh_path(model, 0)
                    vertices, faces = load_mesh(sample_mesh_path)
                    num_vertex = vertices.shape[0]

                    model_color_path = conf.model_color_path(model)
                    model_color = load_model_color(model_color_path)

                    if generate_feature:
                        model_feature = generate_model_feature(params, swi_range, dis_range, rot_range, sess, input, net.feature, num_vertex)
                    else:
                        model_feature_path = conf.model_feature_path(model, params)
                        model_feature = np.load(model_feature_path)

                    print("Prepare key points...")
                    num_clas = 64
                    centroids, belong, _ = furthest_point_sample(vertices, faces, num_clas, 5)
                    key_features = model_feature[centroids]
                    clas_vertices = []
                    clas_features = []
                    for i in range(num_clas):
                        clas_mask = belong == centroids[i]
                        clas_v = np.arange(num_vertex)[clas_mask]
                        clas_vertices.append(clas_v)
                        clas_features.append(model_feature[clas_v])

                    if predict_model:
                        cv.imshow('correspond', np.zeros([512, 512]))
                        cv.imshow('error', np.zeros([512, 512]))
                        for mesh in range(conf.num_meshes[model]):
                            for swi in swi_range:
                                for dis in dis_range:
                                    for rot in rot_range:
                                        view_name = conf.view_name(swi, dis, rot)
                                        depth_view_path = conf.depth_view_path(model, mesh, view_name)
                                        depth_img = cv.imread(depth_view_path, -1)
                                        feature = sess.run(net.feature, feed_dict={input: depth_img.reshape([1, 512, 512, 1])}).reshape([512, 512, 16])

                                        vertex_view_path = conf.vertex_view_path(model, mesh, view_name)
                                        vertex_view = cv.imread(vertex_view_path, -1)
                                        vertex = image_color2idx(vertex_view, rgb=False)

                                        mask = vertex > 0
                                        reff = vertex[mask] - 1

                                        t = timeit()
                                        match = correspond(model_feature, feature, mask)
                                        #match = correspond2(key_features, clas_features, clas_vertices, features, mask)
                                        t = timeit(t)

                                        correspond_img = np.zeros([512, 512, 3], np.uint8)
                                        correspond_img[mask] = model_color[match]

                                        error = np.sqrt(np.sum((vertices[reff] - vertices[match]) ** 2, -1))
                                        mean_err = np.mean(error)
                                        print(mean_err)

                                        error = (error * 1000).astype(np.int32)
                                        error[error >= 200] = 199
                                        error_img = np.zeros([512, 512, 3], np.uint8)
                                        error_img[mask] = err_colors[error]

                                        groundtruth_img = np.zeros([512, 512, 3], np.uint8)
                                        groundtruth_img[mask] = model_color[reff]
                                        cv.imshow('groundtruth', groundtruth_img)
                                        cv.imshow('correspond', correspond_img)
                                        cv.imshow('error', error_img)
                                        cv.waitKey(0)

                    if predict_kinect:
                        cv.imshow('correspond', np.zeros([512, 512]))
                        cv.waitKey(0)
                        for view_idx in range(774, 1000):
                            #depth_view_path = 'data/kinect/kinect_{}.png'.format(str(view_idx).zfill(5))
                            depth_view_path = 'kinect_{}.png'.format(str(774).zfill(5))
                            depth_img = cv.imread(depth_view_path, 0)

                            t = timeit()
                            feature = sess.run(net.feature, feed_dict={input: depth_img.reshape([1, 512, 512, 1])}).reshape([512, 512, 16])
                            
                            mask = depth_img > 0
                            feature = smooth(feature, mask)
                            match = correspond(model_feature, feature, mask)
                            #match = correspond2(key_features, clas_features, clas_vertices, feature, mask)

                            correspond_img = np.zeros([512, 512, 3], np.uint8)
                            correspond_img[mask] = model_color[match]

                            cv.imshow('correspond', correspond_img)
                            #cv.imshow('error', error_img)
                            cv.waitKey(0)
                            timeit(t)
