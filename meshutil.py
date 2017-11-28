from config import *
from plyfile import PlyData, PlyElement
import numpy as np
import gl.glm as glm

def sqr_dist(src, dst):
    sqr_dists = np.multiply(np.dot(dst, src.T), -2)
    sqr_dists += np.sum(np.square(dst), 1, keepdims=True)
    sqr_dists += np.sum(np.square(src), 1)
    return sqr_dists

def load_ply_mesh(mesh_path):
    data = PlyData.read(mesh_path)

    vertex_data = data['vertex'].data
    vertices = np.zeros([vertex_data.shape[0], 3], dtype=np.float32)
    for i in range(vertices.shape[0]):
        for j in range(3):
            vertices[i, j] = vertex_data[i][j]

    face_data = data['face'].data
    faces = np.zeros([face_data.shape[0], 3], dtype=np.int32)
    for i in range(faces.shape[0]):
        for j in range(3):
            faces[i, j] = face_data[i][0][j]

    return vertices, faces

def load_obj_mesh(mesh_path):
    vertex_data = []
    face_data = []
    for line in open(mesh_path, "r"):
        if line.startswith('#'):
            continue
        values = line.split()
        if not values:
            continue
        if values[0] == 'v':
            v = list(map(float, values[1:4]))
            vertex_data.append(v)
        elif values[0] == 'f':
            f = list(map(lambda x: int(x.split('/')[0]),  values[1:4]))
            face_data.append(f)
    vertices = np.array(vertex_data)
    faces = np.array(face_data)
    return vertices, faces

def load_mesh(mesh_path):
    if mesh_path.endswith('.ply'):
        vertices, faces = load_ply_mesh(mesh_path)
    elif mesh_path.endswith('.obj'):
        vertices, faces = load_obj_mesh(mesh_path)
    if np.min(faces) == 1:
       faces -= 1
    vertices = vertices.astype(np.float32)
    faces = faces.astype(np.int32)
    return vertices, faces

def regularize_mesh_old(vertices, model):
    tmp = np.ones([vertices.shape[0], 4], dtype=np.float32)
    tmp[:,:3] = vertices

    if model.startswith('SCAPE'):
        m = glm.identity()
        m = glm.rotate(m, glm.radians(90), glm.vec3(0, 0, 1))
        m = glm.rotate(m, glm.radians(270), glm.vec3(1, 0, 0))
        tmp = glm.transform(tmp, m)
    elif model.startswith('MIT'):
        m = glm.identity()
        m = glm.rotate(m, glm.radians(90), glm.vec3(0, 1, 0))
        tmp = glm.transform(tmp, m)

    vertices[:,:] = tmp[:,:3]

    mean = np.mean(vertices, 0)
    vertices -= mean

def regularize_mesh(vertices, flipyz):
    if flipyz:
        vertices[:,1], vertices[:,2] = vertices[:,2], vertices[:,1]
    
    scale = 1.8 / (np.max(vertices[:,1]) - np.min(vertices[:,1]))
    transform = -np.mean(vertices, 0)
    vertices = (vertices + transform) * scale
    return vertices

def furthest_point_sample(vertices, faces, N, K):
    num_vertices = vertices.shape[0]
    center_indices = np.random.choice(num_vertices, N, replace=False)
    sqr_dists = 1e10 * np.ones(num_vertices)
    vertex_as = np.zeros(num_vertices, dtype=np.int32)
    for i in range(N):
        new_sqr_dists = np.sum(np.square(vertices - vertices[center_indices[i]]), 1)
        update_mask = new_sqr_dists < sqr_dists
        sqr_dists[update_mask] = new_sqr_dists[update_mask]
        vertex_as[update_mask] = i
        next_center = np.argmax(sqr_dists)
        if K - 1 <= i < N - 1:
            center_indices[i + 1] = next_center

    centers = vertices[center_indices]
    face_centers = np.mean(vertices[faces], 1)
    sqr_dists = sqr_dist(centers, face_centers)
    face_as = np.argmin(sqr_dists, 1)
    return center_indices, vertex_as, face_as