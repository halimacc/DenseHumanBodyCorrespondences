from config import *
from meshutil import *
from colorutil import *
import gl.glm as glm
from gl.glrender import *
import time
import cv2 as cv

def gen_segmentation(model, seg_idx, classes):
    sample_mesh_path = conf.mesh_path(model, 0)
    vertices, faces = load_mesh(sample_mesh_path)
    centers, vertex_as, face_as = furthest_point_sample(vertices, faces, classes, 10)

    datas = [centers, vertex_as, face_as]
    types = ['center', 'vertex_as', 'face_as']
    for i in range(3):
        seg_path = conf.segmentation_path(model, seg_idx, types[i])
        np.save(seg_path, datas[i])

def gen_label(model_range, seg_range, swi_range, dis_range, rot_range, depth_and_vertex=True):
    b = conf.zfar * conf.znear / (conf.znear - conf.zfar)
    a = -b / conf.znear

    # preparation
    if depth_and_vertex:
        seg_range = [-1] + seg_range
    seg_color_map = distinct_colors(conf.num_classes)

    renderer = GLRenderer(b'gen_label', (conf.width, conf.height), (0, 0), toTexture=True)
    proj = glm.perspective(glm.radians(70), 1.0, conf.znear, conf.zfar)
    for model in model_range:
        vertex_color = None
        for mesh_idx in range(conf.num_meshes[model]):
            print('Generate label for model {} Mesh {}...'.format(model, mesh_idx))
            mesh_path = conf.mesh_path(model, mesh_idx)
            vertices, faces = load_mesh(mesh_path)
            regularize_mesh(vertices, model)
            faces = faces.reshape([faces.shape[0] * 3])
            vertex_buffer = vertices[faces]

            if vertex_color is None:
                vertex_color = distinct_colors(vertices.shape[0])
            vertex_color_buffer = (vertex_color[faces] / 255.0).astype(np.float32)

            for seg_idx in seg_range:
                # prepare segmentation color
                if seg_idx != -1:
                    seg_path = conf.segmentation_path(model, seg_idx)
                    segmentation = np.load(seg_path)
                    seg_color_buffer = np.zeros([faces.shape[0], 3], np.float32)
                    face_colors = seg_color_map[segmentation] / 255.0
                    seg_color_buffer[2::3,:] = seg_color_buffer[1::3,:] = seg_color_buffer[0::3,:] = face_colors

                for swi in swi_range:
                    for dis in dis_range:
                        for rot in rot_range:
                            mod = glm.identity()
                            mod = glm.rotate(mod, glm.radians(swi - conf.max_swi / 2), glm.vec3(0, 1, 0))
                            mod = glm.translate(mod, glm.vec3(0, 0, -dis / 100.0))
                            mod = glm.rotate(mod, glm.radians(rot), glm.vec3(0, 1, 0))
                            mvp = proj.dot(mod)

                            view_name = conf.view_name(swi, dis, rot)
                            if seg_idx == -1:
                                rgb, z = renderer.draw(vertex_buffer, vertex_color_buffer, mvp.T)
                                # save depth view
                                depth = ((conf.zfar - b / (z - a)) / (conf.zfar - conf.znear) * 255).astype(np.uint8)
                                depth_view_path = conf.depth_view_path(model, mesh_idx, view_name)
                                cv.imwrite(depth_view_path, depth)
                                # save vertex view
                                vertex_view_path = conf.vertex_view_path(model, mesh_idx, view_name)
                                cv.imwrite(vertex_view_path, rgb)
                            else:
                                rgb, z = renderer.draw(vertex_buffer, seg_color_buffer, mvp.T)
                                # save segmentation view
                                seg_view_path = conf.segmentation_view_path(model, mesh_idx, seg_idx, view_name)
                                cv.imwrite(seg_view_path, rgb)

if __name__ == '__main__':
    model_range = ['SCAPE', 'MITcrane', 'MITjumping', 'MITmarch1', 'MITsquat1', 'MITsamba', 'MITswing']

   #gen_segmentation('SCAPE', 101, 64)

    # model_range = ['SCAPE']
    # for model in model_range:
    #     for i in range(5, 100):
    #         gen_segmentation(model, i, conf.num_classes)

    seg_range = [i for i in range(5, 100)]
    swi_range = [25, 35, 45]
    dis_range = [250, 200]
    rot_range = [i for i in range(0, 360, 15)]
    gen_label(model_range, seg_range, swi_range, dis_range, rot_range, True)
