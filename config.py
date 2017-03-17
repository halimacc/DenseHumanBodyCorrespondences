from os import path
from plyfile import PlyData
from objloader import OBJ
import numpy as np
import os

data_dir = 'D:/Project/DHBC/data/'
mesh_dir = data_dir + 'mesh/'
segment_dir = data_dir + 'segmentation/'
view_dir = data_dir + 'view/'

num_model = 11
num_seg = 100
num_patch = 500
num_mesh = [72, 175, 150, 250, 250, 175, 175, 250, 250, 150, 175]

def get_mesh_path(model_idx, mesh_idx):
    if model_idx == 0:
        return mesh_dir + '0/mesh' + str(mesh_idx).zfill(3) + '.ply'
    else:
        return mesh_dir + str(model_idx) + '/mesh_' + str(mesh_idx).zfill(4) + '.obj'


def get_segment_path(model_idx, segment_idx):
    return segment_dir + 'model_{0}_seg_{1}.seg'.format(str(model_idx).zfill(2), str(segment_idx).zfill(3))


# return vetices, faces of a mesh
def load_mesh(mesh_file):
    if path.exists(mesh_file):
        if mesh_file.endswith('.ply'):
            mesh = PlyData.read(mesh_file)
            return mesh['vertex'].data, mesh['face'].data
        elif mesh_file.endswith('.obj'):
            obj = OBJ(mesh_file)
            meshes = np.array(obj.vertices)
            faces = np.array([[idx - 1 for idx in  f[0]] for f in obj.faces])
            return meshes, faces

    return None, None


def int2color(idx):
    idx = idx + 1
    r = idx / 100 / 5.0
    g = idx % 100 / 10 / 10.0
    b = idx % 10 / 10.0
    return (r, g, b)


znear = 1.0
zfar = 3.55

b = zfar * znear / (znear - zfar)
a = - b / znear
def z2depth(z):
    return b / (z - a)

def depth2gray(depth):
    return np.uint8(255 * (zfar - depth) / (zfar - znear))


def get_view_path(model_idx, segment_idx, mesh_idx, view_idx):
    return (os.path.join(view_dir,
                         str(model_idx),
                         str(segment_idx).zfill(3),
                         str(mesh_idx).zfill(3),
                         'view_{0}_color.png'.format(str(view_idx).zfill(3))),
            os.path.join(view_dir,
                         str(model_idx),
                         str(segment_idx).zfill(3),
                         str(mesh_idx).zfill(3),
                         'view_{0}_depth.png'.format(str(view_idx).zfill(3))))


def create_dir(path):
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)