# simple operations that change numpy array
from collections import Counter
from functools import partial
import itertools

import numpy as np
from skimage.measure import label
from scipy.ndimage.morphology import binary_fill_holes


### Operation applied to numpy array

def pad(array, size=1, color=0):
    # size in (1, 10), color: int
    shape = array.shape
    new_array = np.full((shape[0] + 2 * size, shape[1] + 2 * size), color)
    new_array[size:-size, size:-size] = array
    return new_array


def recolor(array, old_color, new_color):
    array_ = array.copy()
    array_[array_ == old_color] = new_color
    return array_


def crop_without_bg(array, bg_color):
    # crop rectangle with content
    array_ = array.copy()
    mask = (array_ != bg_color).astype(int)
    args = np.argwhere(mask > 0)
    axis0_min = np.min(args, axis=0)[0]
    axis1_min = np.min(args, axis=0)[1]
    axis0_max = np.max(args, axis=0)[0]
    axis1_max = np.max(args, axis=0)[1]
    array_ = array_[axis0_min: axis0_max + 1, axis1_min: axis1_max + 1]
    return array_


def repeat(array, n0, n1):
    # n0 in range(2, 10)
    # n1 in range(2, 10)
    return np.tile(array, (n0, n1))


def rotate(array, angle=1):
    # angle in (1,2,3)
    return np.rot90(array, angle)


def flip(array, axis=0):
    # axis in (0, 1)
    return np.flip(array, axis)


def flip_diagonal(array, axis=0):
    # axis in (0, 1)
    if axis == 0:
        return np.swapaxes(array, 0, 1)
    elif axis == 1:
        return np.rot90(np.flip(array, 0), 1)


def repeat_map(array, map):
    # map consists from 0 and 1
    k = array.shape[0]
    l = array.shape[1]
    shape = (map.shape[0] * k, map.shape[1] * l)
    array_ = np.zeros(shape)
    for i in range(map.shape[0]):
        for j in range(map.shape[1]):
            if map[i, j] == 1:
                array_[i * k : (i + 1) * k, j * l : (j + 1) * l] = array
    return array_


def zoom(array, n0, n1):
    array_ = np.kron(array, np.ones((n0, n1)))
    return array_


def identity(array):
    return array


### reversed functions


def reverse_zoom(array, n0, n1):
    shape = array.shape
    k0 = shape[0] // n0
    k1 = shape[1] // n1
    array_ = np.empty((k0, k1))
    for i in range(k0):
        for j in range(k1):
            array_[i, j] = array[i * n0, j * n1]
    return array_


def reverse_repeat(array, n0, n1):
    s0, s1 = array.shape
    k0 = s0 // n0
    k1 = s1// n1
    return array[:k0, : k1]


#### Brute force

range10 = tuple(range(1, 11))
range4 = tuple(range(1, 5))
simple_funcs = [identity, rotate, flip, flip_diagonal, zoom, repeat]
reversed_simple_funcs = [identity, rotate, flip, flip_diagonal, reverse_zoom, reverse_repeat]
simple_func_param_dicts = [{}, {'angle': (1,2,3)}, {'axis': (0,1)}, {'axis': (0, 1)}] + [{'n0': range4, 'n1': range4}] + [{'n0': range4, 'n1': range4}]
reversed_simple_func_param_dicts = [{}, {'angle': (-1,-2,-3)}, {'axis': (0,1)}, {'axis': (0, 1)}] + [{'n0': range10, 'n1': range10}] + [{'n0': range10, 'n1': range10}]


def generate_param_options(param_dict):
    keys = list(param_dict.keys())
    param_values = list(param_dict.values())
    param_options =  list(itertools.product(*param_values))
    param_option_dicts = [{key: value for key, value in zip(keys, param_option)} for param_option in param_options]
    return param_option_dicts


def get_partials(funcs, param_dicts):
    options = []
    for func, param_dict in zip(funcs, param_dicts):
        param_option_dicts = generate_param_options(param_dict)
        func_options = [partial(func, **param_option_dict) for param_option_dict in param_option_dicts]
        options.extend(func_options)
    return options


simple_output_process_options = get_partials(simple_funcs, simple_func_param_dicts)
reversed_simple_output_process_options = get_partials(reversed_simple_funcs, reversed_simple_func_param_dicts)

### Operation detectors

# use output only
def detect_true(array):
    return True


def get_divisors(n):
    divisors = []
    for i in range(1, n//2 + 1):
        if n % i == 0:
            divisors.append(i)
    return divisors + [n]


def detect_zoom(array, n0, n1):
    s0, s1 = array.shape
    axis0_divisors = get_divisors(array.shape[0])
    axis1_divisors = get_divisors(array.shape[1])
    if n0 not in axis0_divisors:
        return False
    if n1 not in axis1_divisors:
        return False
    if n0 == 1 and n1 == 1:
        return False
    if n0 == array.shape[0] and n1 == array.shape[1]:
        return False
    subarrays = get_equal_shape_subarrays(array, n0, n1)
    for subarray in subarrays:
        if len(all_colors(subarray)) != 1:
            return False
    return True


def detect_repeat(array, n0, n1):
    s0, s1 = array.shape
    axis0_divisors = get_divisors(s0)
    axis1_divisors = get_divisors(s1)
    if n0 not in axis0_divisors:
        return False
    if n1 not in axis1_divisors:
        return False
    if n0 == 1 and n1 == 1:
        return False
    if n0 == s0 and n1 == s1:
        return False
    subarrays = get_equal_shape_subarrays(array, s0 // n0, s1 // n1)
    if len(unique_arrays(subarrays)) != 1:
        return False
    return True

# while doing brute force of simple functions run operation only if detect func returns True
detect_simple_functions = [detect_true] * 8 + get_partials([detect_zoom], [{'n0': range10, 'n1': range10}]) + get_partials([detect_repeat], [{'n0': range10, 'n1': range10}])



### SKIMAGE OBJECTS

def get_objects_from_map_(object_map, touch='wall'):
    if touch == 'wall':
        connectivity = 1
    elif touch == 'corner':
        connectivity = 2
    label_array, max_label_index = label(object_map, connectivity=connectivity,
                                         background=0, return_num=True)
    return [(label_array == i).astype(int) for i in range(1, max_label_index + 1)]


def get_most_common_array_color(array):
    colors, color_counts = np.unique(array, return_counts=True)
    most_common_color = colors[np.argmax(color_counts)]
    return most_common_color


def touchs_boundary(map):
    sums = [np.count_nonzero(map[0, 1:-1]),  np.count_nonzero(map[-1, 1:-1]), np.count_nonzero(map[1:-1, 0]), np.count_nonzero(map[1:-1, -1])]
    return sum([s > 0 for s in sums]) >= 3


def detect_bg_(array, how='touch'):
    # TODO: not sure that  i need detect_bg
    if how == 'touch':
        bg = 0
        colors = all_colors(array)
        most_common_color = get_most_common_array_color(array)
        for color in [most_common_color] + list(colors - {most_common_color}):
            object_maps = get_objects_from_map_((array == color).astype(int), touch='corner')
            for object_map in object_maps:
                if touchs_boundary(object_map):
                    bg = color
                    break
    elif how == 'black':
        bg = 0
    elif how == 'most_common':
        bg = get_most_common_array_color(array)
    return bg


def get_objects_by_connectivity_(array, touch='wall', bg_color=None):
    if bg_color is None:
        bg_color = detect_bg_(array)
    array_map = (array != bg_color).astype(int)
    object_maps = get_objects_from_map_(array_map, touch=touch)
    return object_maps


def get_objects_by_color_and_connectivity_(array, touch='wall', bg_color=None):
    if bg_color is None:
        bg_color = detect_bg_(array)
    colors = all_colors(array) - {bg_color}
    object_maps = []
    for color in colors:
        color_object_maps = get_objects_from_map_((array == color).astype(int), touch=touch)
        object_maps.extend(color_object_maps)
    return object_maps


def get_objects_by_color_(array, bg_color=None):
    if bg_color is None:
        bg_color = detect_bg_(array)
    object_maps = []
    colors = all_colors(array) - {bg_color}
    for color in colors:
        object_map = (array == color).astype(int)
        object_maps.append(object_map)
    return object_maps



def get_objects_rectangles(array, direction='vertical', bg_color=None):
    # rectangles of same color can be attached to each other #
    s0, s1 = array.shape
    if bg_color is None:
        bg_color = detect_bg_(array)
    object_maps = []
    #colors = all_colors(array) - {bg_color}
    if direction == 'horisontal':
        object_map = np.zeros_like(array)
        prev_slice = np.full((s1,), -100)
        for i in range(s0):
            slice = array[i]
            if (np.unique(slice) != np.array(bg_color)).any():
                if (slice != prev_slice).any():
                    if np.sum(object_map) != 0:
                        object_maps.append(object_map)
                        object_map = np.zeros_like(array)
                object_map[i] = slice
            else:
                if np.sum(object_map) != 0:
                    object_maps.append(object_map)
                    object_map = np.zeros_like(array)
            prev_slice = slice
        if np.sum(object_map) != 0:
            object_maps.append(object_map)
    elif direction == 'vertical':
        object_map = np.zeros_like(array)
        prev_slice = np.full((s0,), -100)
        for i in range(s1):
            slice = array[:, i]
            if (np.unique(slice) != np.array(bg_color)).any():
                if (slice != prev_slice).any():
                    if np.sum(object_map) != 0:
                        object_maps.append(object_map)
                        object_map = np.zeros_like(array)
                object_map[:, i] = slice
            else:
                if np.sum(object_map) != 0:
                    object_maps.append(object_map)
                    object_map = np.zeros_like(array)
            prev_slice = slice
        if np.sum(object_map) != 0:
            object_maps.append(object_map)
    object_maps = [(object_map > 0).astype(int) for object_map in object_maps]
    return object_maps


def fit_largest_rectangle(object_map):
    largest_rectangle_map = np.zeros_like(object_map)

    nrows, ncols = object_map.shape
    skip = 0
    area_max = (0, [])

    a = object_map
    w = np.zeros(dtype=int, shape=a.shape)
    h = np.zeros(dtype=int, shape=a.shape)
    for r in range(nrows):
        for c in range(ncols):
            if a[r][c] == skip:
                continue
            if r == 0:
                h[r][c] = 1
            else:
                h[r][c] = h[r - 1][c] + 1
            if c == 0:
                w[r][c] = 1
            else:
                w[r][c] = w[r][c - 1] + 1
            minw = w[r][c]
            for dh in range(h[r][c]):
                minw = min(minw, w[r - dh][c])
                area = (dh + 1) * minw
                if area > area_max[0]:
                    area_max = (area, [(r - dh, c - minw + 1, r, c)])
    axis0_min, axis1_min, axis0_max, axis1_max = area_max[1][0]
    largest_rectangle_map[axis0_min: axis0_max + 1, axis1_min: axis1_max + 1] = 1
    return largest_rectangle_map


def remove_padding(object_map):
    axis0_min, axis0_max, axis1_min, axis1_max = get_object_map_min_max(object_map)
    object_map_ = np.zeros_like(object_map)
    object_map_[axis0_min + 1: axis0_max, axis1_min + 1: axis1_max] = 1
    return object_map_


def get_objects_rectangles_without_noise(array, bg_color=None):
    # get rectangle objects out of noisy background
    object_maps = get_objects_by_color_and_connectivity_(array, touch='wall', bg_color=bg_color)
    object_maps = [object_map for object_map in object_maps if np.count_nonzero(object_map) > 1]
    object_maps = [binary_fill_holes(object_map).astype(int) for object_map in object_maps]
    object_maps = [fit_largest_rectangle(object_map) for object_map in object_maps]
    return object_maps


def get_objects_rectangles_without_noise_without_padding(array, bg_color=None):
    object_maps = get_objects_rectangles_without_noise(array, bg_color=bg_color)
    object_maps = [remove_padding(object_map) for object_map in object_maps]
    return object_maps


def get_equal_shape_subarrays(array, l0, l1):
    # (l0, l1) - shape of subarrays
    # untile operations but parameters are dimensions of new array
    subarrays = []
    s0, s1 = array.shape
    n0 = s0 // l0
    n1 = s1 // l1
    for i in range(n0):
        for j in range(n1):
            subarray = array[i * l0: (i + 1) * l0, j * l1 : (j + 1) * l1]
            subarrays.append(subarray)
    return subarrays


def get_object_parts(object_map):
    shape = object_map.shape
    border_map = np.zeros_like(object_map)
    for i in range(shape[0]):
        subarray = object_map[i]
        args = np.argwhere(subarray == 1)[:, 0]
        if len(args) > 0:
            args_min = np.min(args)
            args_max = np.max(args)
            border_map[i, args_min] = 1
            border_map[i, args_max] = 1

    for j in range(shape[1]):
        subarray = object_map[:, j]
        args = np.argwhere(subarray == 1)[:, 0]
        if len(args) > 0:
            args_min = np.min(args)
            args_max = np.max(args)
            border_map[args_min, j] = 1
            border_map[args_max, j] = 1
    inner_map = (object_map != border_map).astype(int)
    return border_map, inner_map


### Object mask operations


def get_rectangle_map(object_map):
    rectangle_mask = np.zeros_like(object_map)
    axis0_min, axis0_max, axis1_min, axis1_max = get_object_map_min_max(object_map)
    rectangle_mask[axis0_min: axis0_max + 1, axis1_min: axis1_max + 1] = 1
    return rectangle_mask


def get_cropped_map(object_map):
    cropped_mask = crop_without_bg(object_map, bg_color=0)
    return cropped_mask


def get_colored_map(array, object_map):
    colored_map = array.copy()
    colored_map[object_map == 0] = -1
    return colored_map


### Object features

def get_object_map_min_max(object_map):
    args = np.argwhere(object_map > 0)
    axis0_min = np.min(args, axis=0)[0]
    axis1_min = np.min(args, axis=0)[1]
    axis0_max = np.max(args, axis=0)[0]
    axis1_max = np.max(args, axis=0)[1]
    return axis0_min, axis0_max, axis1_min, axis1_max


def get_object_dimensions(object_map):
    cropped_mask = get_cropped_map(object_map)
    return cropped_mask.shape


def get_object_size(object_map):
    return np.sum(object_map)


def get_most_common_color(array, object_map):
    colors, color_counts = np.unique(array[object_map == 1], return_counts=True)
    most_common_color = colors[np.argmax(color_counts)]
    return most_common_color


def get_sorted_colors(array, object_map=None):
    if object_map is not None:
        array = array[object_map == 1]
    colors, color_counts = np.unique(array, return_counts=True)
    return colors(np.argsort(color_counts))


def get_object_colors_cnt(array, object_map):
    return Counter(array[object_map == 1].tolist())


def is_rectangle(object_map):
    rectangle_mask = get_rectangle_map(object_map)
    if (rectangle_mask == object_map).all():
        return True
    return False


### Object class


class BaseObject:
    #  object without nested objects
    def __init__(self, array, object_map):
        self.array = array
        self.object_map = object_map
        self.rectangle_map = get_rectangle_map(object_map)
        self.colored_map = get_colored_map(array, object_map)
        self.cropped_map = get_cropped_map(object_map)
        self.cropped_object = crop_without_bg(self.colored_map, bg_color=-1) # bg_color=-1
        self.size = get_object_size(object_map)
        self.height, self.width = self.cropped_map.shape
        self.area = self.height * self.width
        self.is_rectangle = is_rectangle(object_map)
        self.colors, self.color_counts = np.unique(array[object_map == 1], return_counts=True)
        self.n_colors = len(self.colors)
        #self.unique_colors = set(self.colors)
        self.most_common_color = self.colors[np.argmax(self.color_counts)]
        self.least_common_color = self.colors[np.argmin(self.color_counts)]
        self.axis0_min, self.axis0_max, self.axis1_min, self.axis1_max = get_object_map_min_max(self.object_map)


exclude_feature_names = ['array', 'nested_object', 'nested_objects', 'color_counts', 'object_map', 'colored_map']
min_max_feature_names = ['size', 'height', 'width', 'area', 'n_colors', 'axis0_min', 'axis0_max', 'axis1_min', 'axis1_max',
                         'nested_object_size', 'nested_objects_count']

feature_names = min_max_feature_names + ['cropped_map', 'cropped_object', 'is_rectangle',
                                         'most_common_color', 'least_common_color',
                                         'nested_object_shape', 'nested_object_most_common_color']

class Object(BaseObject):
    # object with nested objects
    def __init__(self, array, object_map):
        super().__init__(array, object_map)
        self.nested_objects = make_base_objects(array, get_nested_objects(array, object_map))
        self.nested_objects_count = len(self.nested_objects)
        if len(set(np.unique(self.colored_map)) - {-1}) > 1: # nested object exists
            self.nested_object = BaseObject(array, get_nested_object(array, object_map, self.most_common_color))
            self.nested_object_size = self.nested_object.size
            self.nested_object_shape = self.nested_object.cropped_map
            self.nested_object_most_common_color = self.nested_object.most_common_color
        else:
            self.nested_object = None
            self.nested_object_size = None
            self.nested_object_shape = None
            self.nested_object_most_common_color = None

    def feature_dict(self):
        # all object features in one dictionary
        feature_dict_ = {}
        for attr, value in self.__dict__.items():
            if attr in feature_names:
                if type(value) == np.ndarray:
                    value_ = array_to_tuple(value)
                else:
                    value_ = value
                feature_dict_[attr] = value_
        return feature_dict_


def make_objects(array, object_maps):
    return [Object(array, object_map) for object_map in object_maps]


def make_base_objects(array, object_maps):
    return [BaseObject(array, object_map) for object_map in object_maps]


def get_nested_objects(array, object_map):
    # get all nested objects separately
    array_ = array.copy()
    array_[object_map == 0] = -1
    nested_object_maps = get_objects_by_color_and_connectivity_(array_, touch='wall', bg_color=-1)
    return nested_object_maps


def get_nested_object(array, object_map, most_common_color):
    # get all nested objects all together even if they are not connected
    # exclude most common color
    array_ = array.copy()
    array_[object_map == 0] = -1
    array_[array_ == most_common_color] = -1
    array_[array_ >= 0] = 1
    array_[array_ == -1] = 0
    nested_object_map = array_ #get_objects_by_color_(array_, bg_color=-1)[0]
    return nested_object_map


# Task class and color functions
class Task():
    def __init__(self, task, idx=None):
        self.task = task
        self.inputs = get_inputs(task)
        self.outputs = get_outputs(task)
        self.test_inputs = get_test_inputs(task)
        self.test_outputs = get_test_outputs(task)
        self.test_pairs = get_test_pairs(task)
        self.pairs = get_pairs(task)
        self.idx = idx


def flatten_list(pred):
    return [p for row in pred for p in row]


def get_outputs(task):
    train_examples = task['train']
    outputs = [ex['output'] for ex in train_examples]
    outputs = [np.array(output) for output in outputs]
    return outputs


def get_inputs(task):
    train_examples = task['train']
    inputs = [ex['input'] for ex in train_examples]
    inputs = [np.array(input) for input in inputs]
    return inputs


def get_pairs(task):
    train_examples = task['train']
    pairs = [(np.array(ex['input']), np.array(ex['output'])) for ex in train_examples]
    return pairs


def get_test_inputs(task):
    test = task['test']
    return [np.array(ex['input']) for ex in test]


def get_test_outputs(task):
    test = task['test']
    if 'output' in test[0].keys():
        return [np.array(ex['output']) for ex in test]
    else:
        return None


def get_test_pairs(task):
    test = task['test']
    if 'output' in test[0].keys():
        return [(np.array(ex['input']), np.array(ex['output'])) for ex in test]
    else:
        return None


def colors(array):
    colors = set(flatten_list(array)) - {0}
    return colors


def all_colors(array):
    colors = set(flatten_list(array))
    return colors


def array_to_tuple(array):
    if len(array.shape) == 2:
        res = tuple(tuple(a.tolist()) for a in array)
    else:
        res = tuple(array.tolist())
    return res


def unique_arrays(arrays):
    arrays = [array_to_tuple(a) for a in arrays]
    arrays = list(set(arrays))
    arrays = [np.array(a) for a in arrays]
    return arrays


def all_task_colors(task):
    arrays = task.inputs + task.outputs + task.test_inputs
    colors = set(flatten_list([all_colors(array) for array in arrays]))
    return colors


def colors_only_in_task_outputs(task):
    input_colors = set(flatten_list([all_colors(array) for array in task.test_inputs]))
    output_colors = set(flatten_list([all_colors(array) for array in task.test_outputs]))
    diff = output_colors - input_colors
    return diff


def colors_only_in_input(input_, output):
    input_colors = all_colors(input_)
    output_colors = all_colors(output)
    return input_colors - output_colors


def colors_only_in_output(input_, output):
    input_colors = all_colors(input_)
    output_colors = all_colors(output)
    return output_colors - input_colors