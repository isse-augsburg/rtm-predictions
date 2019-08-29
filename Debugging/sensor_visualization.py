import h5py
import matplotlib.pyplot as plt
import numpy as np
from os import walk
import os.path
from tqdm import tqdm
from datetime import datetime
import multiprocessing as mp
from functools import partial
import time


# returns list of (id, (x ,y)) of sensors
def parse_out_file(filename):
    f = open(filename, "r")
    lines = f.readlines()

    all_sensors = []

    id_string = "sensor number:"
    coords_string = "sensor location:"
    sensor_index, sensor_coords = 0, (0, 0)

    for line in lines:
        line = line.strip()

        if id_string in line:
            slice = line.split(":")[-1]
            sensor_index = int(slice) - 1

        if coords_string in line:
            slice = line.split(":")[-1].strip()
            slice = slice.split(" ")[:-1]
            slice = [float(x) for x in slice]
            sensor_coords = tuple(slice)
            all_sensors.append([sensor_index, sensor_coords])
            sensor_index, sensor_coords = 0, (0, 0)

    return all_sensors


def get_filelist_within_folder(root_directory, num_filenames=-1):
    dataset_filenames = []

    for dirs in root_directory:

        for (dirpath, _, filenames) in tqdm(walk(dirs)):
            if filenames:
                filenames = [dirpath + '/' + f for f in filenames if f.endswith('.erfh5')]
                dataset_filenames.extend(filenames)
                if num_filenames > 0 and len(dataset_filenames) > num_filenames:
                    break

    return dataset_filenames


def get_all_sensor_values(filename):
    f = h5py.File(filename, 'r')

    pressure_array = \
    f['post']['multistate']['TIMESERIES1']['multientityresults']['SENSOR']['PRESSURE']['ZONE1_set1']['erfblock']['res'][
        ()]
    pressure_array = pressure_array / 100

    last_filling = pressure_array[-1]

    non_zeros = np.count_nonzero(last_filling)
    all_entries = np.shape(last_filling)[0]
    filling = non_zeros / all_entries

    return pressure_array, filling


def __plot_sensorgrid(all_sensors, values, text=""):
    for sensor_id, (x, y) in all_sensors:
        v = int(values[sensor_id])
        # color = np.array([[min(value, 255), 255, 255]])

        color = np.array([[min(max(255 - v, 0), 255), min(max(255 - v, 0), 255), min(max(255 - v, 0), 255)]])
        plt.scatter(x, y, c=color / 255)

    plt.figtext(0.02, 0.02, s=text)
    plt.show()


def plot_sensor_series(filename_data, filename_sensorinfo):
    all_sensors = parse_out_file(filename_sensorinfo)
    sensor_values, percentage = get_all_sensor_values(filename_data)

    last_step = sensor_values[-1]

    for i, v in enumerate(sensor_values):
        if i % 50 == 0:
            __plot_sensorgrid(all_sensors, v, "Filling percentage: " + str(percentage))

    print(">>> Plotting FINAL step")
    __plot_sensorgrid(all_sensors, last_step, "Filling percentage: " + str(percentage) + " || Last step")


def __plot_and_save_sensorstep(file, save_path=None):
    sensor_values, percentage = get_all_sensor_values(file)
    color = np.array([[0, 0, 1.0]])
    save_path_negative = save_path + 'Negative/'
    save_path_positive = save_path + 'Positive/'

    last_step = sensor_values[-1]

    for i, value in enumerate(last_step):
        plt.scatter(i, value, s=1, c=color)
        plt.figtext(0.02, 0.02, s="Filling percentage: " + str(percentage))

    if not save_path is None:
        timestamp = int(datetime.timestamp(datetime.now()))
        fname = str(timestamp) + '.png'

        if percentage < 1:
            plt.savefig(save_path_negative + fname)
        else:
            plt.savefig(save_path_positive + fname)

        plt.clf()
    else:
        plt.show()
        plt.clf()


def plot_all_last_steps(root_directory, save_path=None, num_files=-1, num_workers=-1):
    print("Getting all filenames.")
    # save_path = '/run/user/1001/gvfs/smb-share:server=137.250.170.56,share=home/l/o/lodesluk/Data Analysis/'

    all_files = get_filelist_within_folder([root_directory], num_filenames=num_files)

    save_path_negative = save_path + 'Negative/'
    save_path_positive = save_path + 'Positive/'

    if not os.path.exists(save_path_negative):
        os.makedirs(save_path_negative)

    if not os.path.exists(save_path_positive):
        os.makedirs(save_path_positive)

    num_pos, num_neg = 0, 0

    if num_workers > 0:
        start_time = time.time()
        pool = mp.Pool(processes=num_workers)
        pool.map(partial(__plot_and_save_sensorstep, save_path=save_path), all_files)
        end_time = time.time()
        delta = end_time - start_time
        print("Calculation took:", delta)

    else:
        for f in tqdm(all_files):
            __plot_and_save_sensorstep(f, save_path=save_path)


if __name__ == "__main__":
    filename_sensorinfo = '/run/user/1001/gvfs/smb-share:server=137.250.170.56,share=share/data/RTM/Leoben/output/with_shapes/2019-07-23_15-38-08_5000p/39/2019-07-23_15-38-08_39d.out'
    filename_data = '/run/user/1001/gvfs/smb-share:server=137.250.170.56,share=share/data/RTM/Leoben/output/with_shapes/2019-07-23_15-38-08_5000p/39/2019-07-23_15-38-08_39_RESULT.erfh5'
    # root =  '/run/user/1001/gvfs/smb-share:server=137.250.170.56,share=share/data/RTM/Lautern/output/with_shapes/2019-06-05_15-30-52_1050p/'
    save_path = '/home/lodes/DataAnalysis/Test/'
    root = '/run/user/1001/gvfs/smb-share:server=137.250.170.56,share=share/data/RTM/Leoben/output/with_shapes/2019-07-23_15-38-08_5000p'
    # parse_out_file(filename)
    # plot_sensor_series(filename_data, filename_sensorinfo)
    plot_all_last_steps(root, save_path=save_path, num_files=50, num_workers=6)
