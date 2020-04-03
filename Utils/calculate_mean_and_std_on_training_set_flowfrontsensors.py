import pickle
from pathlib import Path

import numpy as np

import Resources.training as r
from Pipeline import torch_datagenerator as td
from Pipeline.data_gather import get_filelist_within_folder_blacklisted
from Pipeline.data_loader_flowfront_sensor import DataloaderFlowfrontSensor

if __name__ == "__main__":
    dlds = DataloaderFlowfrontSensor(sensor_indizes=((1, 8), (1, 8)))
    generator = td.LoopingDataGenerator(
        r.get_data_paths_debug(),
        get_filelist_within_folder_blacklisted,
        dlds.get_flowfront_sensor_bool_dryspot,
        num_validation_samples=131072,
        num_test_samples=1048576,
        batch_size=131072,
        split_load_path=r.datasets_dryspots,
        split_save_path=Path(),
        num_workers=75,
        looping_strategy=None
    )
    mean = 0.
    std = 0.
    j = 0
    for i, (inputs, _, _) in enumerate(generator):
        abs_speed_at_sensors = np.linalg.norm(inputs, axis=2)
        mean_at_sensors = abs_speed_at_sensors.mean(axis=0)
        mean = mean + mean_at_sensors
        std_at_sensors = abs_speed_at_sensors.std(axis=0)
        std = std + std_at_sensors
        j = i

    _std = std / (j + 1)
    _mean = mean / (j + 1)
    print("Std\n", _std)
    print("Mean\n", _mean)
    pickle.dump((_mean, _std), open("mean_std_20_sensors.p", "wb"))
