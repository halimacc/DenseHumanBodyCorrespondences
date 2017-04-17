from os import path
from plyfile import PlyData
from objloader import OBJ
import numpy as np
import os

data_dir = 'D:\\Project\\DHBC\\data\\'
mesh_dir = data_dir + 'mesh\\'
segment_dir = data_dir + 'segmentation\\'
view_dir = data_dir + 'view\\'

num_model = 11
#num_seg = 100
num_seg =  4
num_patch = 500
num_mesh = [71, 175, 150, 250, 250, 175, 175, 250, 250, 150, 175]
num_dense_color = 500


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
            faces = np.array([[idx - 1 for idx in f[0]] for f in obj.faces])
            return meshes, faces

    return None, None

rgbRange = int(256 * 256 * 256)
def int2color(integer, max=rgbRange):
    integer = integer * (rgbRange // max)
    r = (integer // 256 // 256) / 255.0
    g = (integer // 256 % 256) / 255.0
    b = (integer % 256) / 255.0
    return (r, g, b)

def color2int(color, max=rgbRange):
    r, g, b = color[0], color[1], color[2]
    integer = 255 * 255 * r + 255 * g + b;
    integer = integer // (rgbRange // max);
    return integer

znear = 1.0
zfar = 3.55

b = zfar * znear / (znear - zfar)
a = - b / znear

def z2depth(z):
    return b / (z - a)


def depth2gray(depth):
    return np.uint8(255 * (zfar - depth) / (zfar - znear))

def get_depth_view_path(model_idx, mesh_idx, view_idx):
    return view_dir + "model_{0}\\mesh_{1}\\depth\\{2}.png".format(str(model_idx).zfill(2),
                            str(mesh_idx).zfill(3),
                            str(view_idx).zfill(3))
    
def get_vertex_view_path(model_idx, mesh_idx, view_idx):
    return view_dir + "model_{0}\\mesh_{1}\\vertex\\{2}.png".format(str(model_idx).zfill(2),
                            str(mesh_idx).zfill(3),
                            str(view_idx).zfill(3))

def get_segmentation_view_path(model_idx, mesh_idx, segmentation_idx, view_idx):
    return view_dir + "model_{0}\\mesh_{1}\\segmentation_{2}\\{3}.png".format(str(model_idx).zfill(2),
                            str(mesh_idx).zfill(3),
                            str(segmentation_idx).zfill(3),
                            str(view_idx).zfill(3))
    
def get_pred_name(model_idx, segmentation_idx):
    return 'pred_model_{0}_segmentation_{1}'.format(str(model_idx).zfill(2), str(segmentation_idx).zfill(3))

def create_dir(path):
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
        
def f2u(color):
    c = []
    for i in range(3):
        c.append(255 if color[i] == 1 else int(256 * color[i]))
    return (c[0], c[1], c[2])

#print(rgb2int)
if __name__ == '__main__':
    for i in range(180, 200):
        color = int2color(i)
        rgb = f2u(color)
        print(i, color, rgb)