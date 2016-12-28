from os import path
from os import path
from plyfile import PlyData

data_dir = '/home/chong/data/dhbc/'
mesh_dir = data_dir + 'mesh/'
segment_dir = data_dir + 'segment/'

num_model = 1
num_segment = 500


def get_mesh_path(model_idx, mesh_idx):
    if model_idx == 0:
        return mesh_dir + '0/mesh' + str(mesh_idx).zfill(3) + '.ply'
    else:
        return mesh_dir + str(model_idx) + '/mesh_' + str(mesh_idx).zfill(4) + '.obj'


def get_segment_path(model_idx, segment_idx):
    return segment_dir + str(model_idx) + '/segment_' + str(segment_idx).zfill(3) + '.seg'


# return vetices, faces of a mesh
def load_mesh(mesh_file):
    if path.exists(mesh_file):
        if mesh_file.endswith('.ply'):
            mesh = PlyData.read(mesh_file)
            return mesh['vertex'].data, mesh['face'].data
        elif mesh_file.endswith('.obj'):
            obj = OBJ(mesh_file)

    return None, None
