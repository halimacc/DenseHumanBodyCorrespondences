from config import *
import numpy as np
import os
from furthest_point_sample import fps, assign_faces

print "Start to generate model segmentations..."
for model_idx in xrange(0, 1):#num_model):
    sample_mesh = get_mesh_path(model_idx, 0)
    if not os.path.exists(sample_mesh): 
        continue

    print "Load mesh {0}...".format(sample_mesh)
    vertices, faces = load_mesh(sample_mesh)

    for i in xrange(num_seg):
        print "Generate segmetation {0} for model {1}".format(i, model_idx)

        print "Calculate centroids..."
        centroid_indices = fps(vertices, num_patch, 10)

        print "Assign faces to centroids..."
        face_assignment = assign_faces(faces, vertices, centroid_indices)

        segment_path = get_segment_path(model_idx, i)
        if not os.path.exists(os.path.dirname(segment_path)):
            os.makedirs(os.path.dirname(segment_path))
        print "Save segment to {0}".format(segment_path)
        np.savetxt(segment_path, face_assignment)
