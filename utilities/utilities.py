import os
import random
import string
from pathlib import Path

import numpy
from matplotlib import pyplot


def get_random_num_or_text(size: int = 5, numeric: bool = False):
    # txt = f'{"".join(random.choices(string.ascii_letters, k=size))}'
    # return txt

    return f'{"".join(random.choices(string.ascii_letters, k=size))}' \
        if numeric is False else f'{"".join(random.choices(string.digits, k=size))}'


def get_file_name_without_extension(filename):
    return Path(filename).stem


def get_file_name_extension(filename):
    return Path(filename).suffix


def get_numpy_random_nd_array(lowerint=-10, higherint=10, size: int = 10):
    return numpy.random.randint(lowerint, higherint, size)


def plot_save_graphs(graph_info_tuple: tuple):
    x_dataset, y_dataset, x_label, y_label, graph_title, graph_file_name = graph_info_tuple
    pyplot.plot(x_dataset, y_dataset)
    pyplot.xlabel(x_label)
    pyplot.ylabel(y_label)
    pyplot.title(graph_title)
    folder_path = os.path.dirname(graph_file_name)
    graph_folder = "graphs"
    graph_folder = os.path.join(folder_path, graph_folder)
    if not os.path.exists(graph_folder):
        try:
            os.mkdir(graph_folder)
        except FileExistsError:
            pass
    graph_file_name_wo_ext = get_file_name_without_extension(os.path.join(graph_folder, graph_file_name))
    graph_file_extension = get_file_name_extension(os.path.join(graph_folder, graph_file_name))
    graph_file_name = f'{graph_file_name_wo_ext}_{get_random_num_or_text(5, numeric=True)}{graph_file_extension}'
    graph_file_name = os.path.join(graph_folder, graph_file_name)
    pyplot.savefig(graph_file_name)
    pyplot.close()
