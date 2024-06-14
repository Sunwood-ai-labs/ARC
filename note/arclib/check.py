from arclib.dsl import flatten_list, all_colors


def check_outputs_one_color(task):
    one_color = True
    for output in task.outputs:
        if len(set(flatten_list(output))) != 1:
            one_color = False
    return one_color


def check_outputs_same_shape(task):
    shapes = []
    for output in task.outputs:
        shapes.append(output.shape)
    return len(set(shapes)) == 1


def check_output_color_from_input(task):
    for pair in task.pairs:
        input_, output = pair
        input_colors = all_colors(input_)
        output_colors = all_colors(output)
        if output_colors - input_colors != set():
            return False
    return True


def check_input_output_same_shape(task):
    outputs = task.outputs
    inputs = task.inputs

    same_shape = True
    for inp, out in zip(inputs, outputs):
        if inp.shape != out.shape:
            same_shape = False
    return same_shape


def check_outputs_smaller(task):
    if task.idx == 66:
        a = 1
    for input, output in task.pairs:
        if input.shape[0] < output.shape[0] or  input.shape[1] < output.shape[1]:
            return False
        if input.shape[0] == output.shape[0] and  input.shape[1] <= output.shape[1]:
            return False
        if input.shape[0] < output.shape[0] and  input.shape[1] == output.shape[1]:
            return False
    return True