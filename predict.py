from net import DHBC
from config import conf
import gl.glm as glm
from gl.glrender import GLRenderer
from meshutil import regularize_mesh, load_mesh
from colorutil import distinct_colors, image_color2idx

import argparse
import tensorflow as tf
import cv2 as cv
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def generate_mesh_feature(sess, input, output, mesh_path, flipyz=False):
    b = conf.zfar * conf.znear / (conf.znear - conf.zfar)
    a = -b / conf.znear

    renderer = GLRenderer(b'generate_mesh_feature', (conf.width, conf.height), (0, 0), toTexture=True)
    proj = glm.perspective(glm.radians(70), 1.0, conf.znear, conf.zfar)

    vertices, faces = load_mesh(mesh_path)
    vertices = regularize_mesh(vertices, flipyz)
    faces = faces.reshape([faces.shape[0] * 3])
    vertex_buffer = vertices[faces]
    vertex_color = distinct_colors(vertices.shape[0])
    vertex_color_buffer = (vertex_color[faces] / 255.0).astype(np.float32)

    cnt = np.zeros([vertices.shape[0]], dtype=np.int32)
    feat = np.zeros([vertices.shape[0], 16], dtype=np.float32)

    swi = 35
    dis = 200
    for rot in range(0, 360, 15):
        mod = glm.identity()
        mod = glm.rotate(mod, glm.radians(swi - conf.max_swi / 2), glm.vec3(0, 1, 0))
        mod = glm.translate(mod, glm.vec3(0, 0, -dis / 100.0))
        mod = glm.rotate(mod, glm.radians(rot), glm.vec3(0, 1, 0))
        mvp = proj.dot(mod)

        rgb, z = renderer.draw(vertex_buffer, vertex_color_buffer, mvp.T)

        depth = ((conf.zfar - b / (z - a)) / (conf.zfar - conf.znear) * 255).astype(np.uint8)
        features = sess.run(output, feed_dict={input: depth.reshape([1, 512, 512, 1])}).reshape([512, 512, 16])

        vertex = image_color2idx(rgb)

        mask = vertex > 0
        vertex = vertex[mask]
        features = features[mask]
        for i in range(vertex.shape[0]):
            cnt[vertex[i] - 1] += 1
            feat[vertex[i] - 1] += features[i]

    for i in range(vertices.shape[0]):
        if cnt[i] > 0:
            feat[i] /= cnt[i]
    return feat


if __name__ == '__main__':
    # initial parser
    parser = argparse.ArgumentParser(description='Dense Human Body Correspondences TensorFlow implementation.')
    parser.add_argument('--checkpoint', type=str, help='path of model checkpoint', required=True)
    parser.add_argument('--output', type=str, help='path of feature to store', required=True)
    parser.add_argument('--depth', type=str, help='path of depth image to predict')
    parser.add_argument('--mesh', type=str, help='path of 3D human mesh to predict, support OBJ and PLY now')
    parser.add_argument('--flipyz', help='flip y and z axis of mesh', action='store_true')
    
    args = parser.parse_args()

    print('Initialize network...')
    tf.Graph().as_default()
    dhbc = DHBC()
    input = tf.placeholder(tf.float32, [1, None, None, 1])
    feature = dhbc.forward(input)

    print('Load checkpoit from {}...'.format(args.checkpoint))
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(dhbc.feat_vars)
    saver.restore(sess, args.checkpoint)

    if args.depth:
        print('Extract feature for depth image {}...'.format(args.depth))
        depth_img = cv.imread(args.depth, -1)
        h, w = depth_img.shape[:2]
        depth_img = depth_img.reshape([1, h, w, 1])

        output = sess.run(feature, feed_dict={input: depth_img})[0]

        np.save(args.output, output)        
        print('Save feature at {}, formated as [{}, {}, 16] float numpy array.'.format(args.output, h, w))
    elif args.mesh:
        print('Extract feature for mesh {}...'.format(args.mesh))
        output = generate_mesh_feature(sess, input, feature, args.mesh, args.flipyz)

        np.save(args.output, output)
        print('Save feature at {}, formated as [vertex_num, 16] float numpy array.'.format(args.output))