import pickle
from pathlib import Path

import torch

import Resources.training as r
from Pipeline import torch_datagenerator as td
from Pipeline.data_gather import get_filelist_within_folder_blacklisted
from Pipeline.data_loader_dryspot import DataloaderDryspots

if __name__ == "__main__":
    dlds = DataloaderDryspots(divide_by_100k=False)
    batch_size = 131072
    generator = td.LoopingDataGenerator(
        r.get_data_paths_base_0(),
        get_filelist_within_folder_blacklisted,
        dlds.get_sensor_bool_dryspot,
        num_validation_samples=131072,
        num_test_samples=1048576,
        batch_size=batch_size,
        split_load_path=r.datasets_dryspots,
        split_save_path=Path(),
        num_workers=75,
        looping_strategy=None
    )
    all_sensor_inputs = []
    for i, (inputs, _, _) in enumerate(generator):
        all_sensor_inputs.append(inputs)
        print(i)
    all_sensor_values = torch.cat(all_sensor_inputs, dim=0)
    _std = all_sensor_values.std(dim=0)
    _mean = all_sensor_values.mean(dim=0)
    print("Std\n", _std)
    print("Mean\n", _mean)
    pickle.dump((_mean, _std), open("mean_std_1140_pressure_sensors.p", "wb"))
