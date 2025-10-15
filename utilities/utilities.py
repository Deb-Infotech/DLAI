import os
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend
import numpy as np
from matplotlib import pyplot


def get_current_date_time_stamp():
    dt_time: datetime = datetime.now()
    return dt_time.strftime("%d_%m_%y_%H_%M_%S") + f'_{dt_time.microsecond // 1000:03d}'


def get_file_name_without_extension(filename):
    return Path(filename).stem


def get_file_name_extension(filename):
    return Path(filename).suffix


def get_numpy_random_nd_array(lowerint=-10, higherint=10, size: int = 10):
    return np.sort(np.random.choice(range(lowerint, higherint), size=size, replace=False))


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
    graph_file_name = f'{graph_file_name_wo_ext}_{get_current_date_time_stamp()}{graph_file_extension}'
    graph_file_name = os.path.join(graph_folder, graph_file_name)
    pyplot.savefig(graph_file_name)
    pyplot.close()
