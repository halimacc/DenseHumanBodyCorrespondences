from config import *
import numpy as np

def idx2color(idx):
    r = idx // (256 * 256) % 256
    g = idx // 256 % 256
    b = idx % 256
    return np.array([r, g, b], dtype=np.uint8)

def image_color2idx(color_img, rgb=False):
    color_img = color_img.astype(np.int32)
    idx = np.zeros([color_img.shape[0], color_img.shape[1]], np.int32)
    if rgb:
        idx[:,:] += color_img[:,:,2] * 256 * 256
        idx[:,:] += color_img[:,:,1] * 256
        idx[:,:] += color_img[:,:,0]
    else:
        idx[:,:] += color_img[:,:,0] * 256 * 256
        idx[:,:] += color_img[:,:,1] * 256
        idx[:,:] += color_img[:,:,2]
    return idx

def image_int2color(int_img, rgb=False):
    color_img = np.zeros([int_img.shape[0], int_img.shape[1], 3], np.uint8)
    color_img[:,:,0] = int_img // (256 * 256) % 256
    color_img[:,:,1] = int_img // 256 % 256
    color_img[:,:,2] = int_img % 256
    return color_img

def distinct_colors(num_classes):
    colors = np.zeros([num_classes, 3], np.uint8)
    for i in range(num_classes):
        colors[i] = idx2color(i + 1)
    return colors