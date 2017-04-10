import config
for model_idx in xrange(11):
    for segmentation_idx in xrange(100):
        for mesh_idx in xrange(config.num_mesh[model_idx]):
            color_path, depth_path = config.get_view_path(model_idx, segmentation_idx, mesh_idx, 0)
            config.create_dir(color_path)
