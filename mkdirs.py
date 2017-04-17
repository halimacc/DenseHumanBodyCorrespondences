import config

print(config.get_depth_view_path(0,0,0))
for model_idx in range(11):
    for mesh_idx in range(config.num_mesh[model_idx]):
        depth_view_path = config.get_depth_view_path(model_idx, mesh_idx, 0)
        config.create_dir(depth_view_path)
        vertex_view_path = config.get_vertex_view_path(model_idx, mesh_idx, 0)
        config.create_dir(vertex_view_path)
        for segmentation_idx in range(100):
            segmentation_view_path = config.get_segmentation_view_path(model_idx, mesh_idx, segmentation_idx, 0)
            config.create_dir(segmentation_view_path)


#import config
#import os
#model_idx = 0
#for segmentation_idx in range(100):
#    os.rmdir(config.view_dir + '{0}/{1}/{2}'.format(str(model_idx), str(segmentation_idx).zfill(3), str(51).zfill(3)))
#    for mesh_idx in range(52, 72):
#        src_folder = config.view_dir + '{0}/{1}/{2}'.format(str(model_idx), str(segmentation_idx).zfill(3), str(mesh_idx).zfill(3))
#        dst_folder = config.view_dir + '{0}/{1}/{2}'.format(str(model_idx), str(segmentation_idx).zfill(3), str(mesh_idx - 1).zfill(3))
#        os.rename(src_folder, dst_folder)
        