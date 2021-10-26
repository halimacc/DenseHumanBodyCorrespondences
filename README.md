Dense Human Body Correspondences (Deprecated)
==================================
**2021-10-25: This repo is no longer under maintenance.** My apologies, should have done this a long time ago.

This is a tensorflow implementation for paper [**Dense Human Body Correspondences Using Convolutional Networks**](https://arxiv.org/abs/1511.05904).

**ATTENTION**: This repo is currently semi-finished, in next few weeks, the newest:
- visualize scripts
- training tutorial

will be updated.

## Installation
1. Clone this repository to your computer.
2. Modify `project_dir` in `config.py` to the path of this repo.
3. Download 3D human model meshes [data.zip](https://pan.baidu.com/s/1bUXSSY) (48M), unzip it to the repo directory. The structure should be like `path/to/repo/data/..`.
4. Download pretrained model [alex-SM-5](https://pan.baidu.com/s/1qYoONuc) (121M), unzip it to the models directory. The structure should be like `path/to/repo/models/alex-SM-5/..`

## Usage
### Predict feature for depth scan
For input depth image with shape [H, W, 1], outputs [H, W, 16] numpy array. Example:

    python predict.py --checkpoint ./models/alex-SM-5/model --output feature.npy --depth ./test.png

### Predict feature for 3D human mesh
For input human mesh model (support `obj` and `ply`), outputs [vertex_count, 16] numpy array. 
The input mesh will be format as 1.8m tall, zero-centerd.

    python predict.py --checkpoint ./models/alex-SM-5/model --output feature.npy --mesh ./test.obj --flipyz
