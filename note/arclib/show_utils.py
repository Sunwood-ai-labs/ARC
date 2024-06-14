
import matplotlib.pyplot as plt
from matplotlib import colors


def plot_array(array):
    cmap = colors.ListedColormap(
        ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
         '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
    norm = colors.Normalize(vmin=0, vmax=9)
    fig, ax = plt.subplots(1)
    ax.imshow(array, cmap=cmap, norm=norm)
    ax.grid(True, which='both', color='lightgrey', linewidth=0.5)
    ax.set_yticks([x - 0.5 for x in range(1 + len(array))])
    ax.set_xticks([x - 0.5 for x in range(1 + len(array[0]))])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.show()


def plot_one(task, ax, i, train_or_test, input_or_output):
    task = task.task
    cmap = colors.ListedColormap(
        ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
         '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
    norm = colors.Normalize(vmin=0, vmax=9)

    input_matrix = task[train_or_test][i][input_or_output]
    #print(input_matrix)
    ax.imshow(input_matrix, cmap=cmap, norm=norm)
    ax.grid(True, which='both', color='lightgrey', linewidth=0.5)
    ax.set_yticks([x - 0.5 for x in range(1 + len(input_matrix))])
    ax.set_xticks([x - 0.5 for x in range(1 + len(input_matrix[0]))])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title(train_or_test + ' ' + input_or_output)
    #plt.show()


def plot_task(task):
    """
    Plots the first train and test pairs of a specified task,
    using same color scheme as the ARC app
    """
    num_train = len(task.inputs)
    fig, axs = plt.subplots(2, num_train, figsize=(3 * num_train, 3 * 2))
    for i in range(num_train):
        plot_one(task, axs[0, i], i, 'train', 'input')
        plot_one(task, axs[1, i], i, 'train', 'output')
    plt.tight_layout()
    plt.show()

    num_test = len(task.test_inputs)
    fig, axs = plt.subplots(2, num_test, figsize=(3 * num_test, 3 * 2))
    if num_test == 1:
        plot_one(task, axs[0], 0, 'test', 'input')
        plot_one(task, axs[1], 0, 'test', 'output')
    else:
        for i in range(num_test):
            plot_one(task, axs[0, i], i, 'test', 'input')
            plot_one(task, axs[1, i], i, 'test', 'output')
    plt.tight_layout()
    plt.show()