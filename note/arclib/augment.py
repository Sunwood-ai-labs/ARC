import numpy as np


def flip_horisontal(array):
    return [array, np.flip(array, 1).copy()]


def flip_vertical(array):
    return [array, np.flip(array, 0).copy()]


def rotate(array, angle=90):
    new_array = array.copy()
    for n in range(int(angle)//90):
        new_array = np.rot90(new_array).copy()
    return [array, new_array]


def all_flip_rotate(array):
    arrays = [array]
    rotate_arrays = [rotate(array, angle)[1] for angle in [90, 180, 270]]
    arrays.extend(rotate_arrays)
    flip_arrays = [np.flip(array, 0).copy() for array in arrays]
    arrays.extend(flip_arrays)
    return arrays


def augment(array, sym=None, colors=False):
    if sym == 'all':
        augmented = all_flip_rotate(array)
    elif sym=='hor':
        augmented = flip_horisontal(array)
    elif sym == 'vert':
        augmented = flip_vertical(array)
    else:
        augmented = [array]
    if colors:
        pass
    return augmented
