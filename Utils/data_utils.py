import hashlib
import json
import os
import re
from pathlib import Path

import h5py
import numpy as np
from scipy import spatial

import Resources.training as r


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def handle_torch_caching(processing_function):
    data_loader_info = processing_function.__self__.__dict__
    data_loader_info["data_processing_function"] = processing_function.__name__
    data_loader_info["data_loader_name"] = processing_function.__self__.__class__.__name__
    print(data_loader_info)
    data_loader_hash = hashlib.md5(str(data_loader_info).encode("utf-8")).hexdigest()
    print(data_loader_hash)
    load_and_save_path = r.datasets_dryspots_torch / data_loader_hash
    load_and_save_path.mkdir(exist_ok=True)

    if (r.datasets_dryspots_torch / "info.json").is_file():
        with open(r.datasets_dryspots_torch / "info.json", "r") as f:
            data = json.load(f)
    else:
        data = {}
    data.update({data_loader_hash: data_loader_info})
    with open(r.datasets_dryspots_torch / "info.json", "w") as f:
        json.dump(data, f, cls=NumpyEncoder)

    return load_and_save_path


def extract_sensor_coords(fn: Path, indices=((0, 1), (0, 1))):
    """
    Extract the sensor coordinates as numpy array from a *d.out file, which exists for every run.
    """
    with fn.open() as f:
        content = f.read()
    sensor_coords = []
    for triple in re.findall(r"\d+\.\d+ \d+\.\d+ \d+\.\d+", content):
        sensor_coords.append([float(e) for e in triple.split(' ')])
    _s_coords = np.array(sensor_coords)
    # Cut off last column (z), since it is filled with 1s anyway
    _s_coords = _s_coords[:, :-1]
    # if indices != ((0, 1), (0, 1)):
    #     _s_coords = _s_coords.reshape(38, 30)
    #     _s_coords = _s_coords[indices[0][0]::indices[0][1], indices[1][0]::indices[1][1]]
    #     _s_coords = _s_coords.flatten()
    return _s_coords


def extract_coords_of_mesh_nodes(fn: Path, normalized=True):
    """
    Extract the coordinates of the mesh nodes as numpy array from a *RESULT.erfh5 file, which exists for every run.
    """
    with h5py.File(fn, 'r') as f:
        coord_as_np_array = f["post/constant/entityresults/NODE/COORDINATE/ZONE1_set0/erfblock/res"][()]
    # Cut off last column (z), since it is filled with 1s anyway
    _coords = coord_as_np_array[:, :-1]
    if normalized:
        _coords = normalize_coords(_coords)
    return _coords


def get_node_propery_at_states(f: h5py.File, node_property: str, states: list):
    return [
        f["post"]["singlestate"][state]["entityresults"]["NODE"][node_property]["ZONE1_set1"]["erfblock"]["res"][()]
        for state in states]


def extract_nearest_mesh_nodes_to_sensors(fn: Path):
    sensor_coords = extract_sensor_coords(Path(str(fn) + "d.out"))
    nodes_coords = extract_coords_of_mesh_nodes(Path(str(fn) + "_RESULT.erfh5"), normalized=False)
    dists_indeces = []
    for sensor in sensor_coords:
        dists_indeces.append(spatial.KDTree(nodes_coords).query(sensor))
    dists, indices = zip(*dists_indeces)
    return np.array(indices)


def scale_coords_lautern(input_coords):
    scaled_coords = (input_coords + 23.25) * 10
    return scaled_coords


def scale_coords_leoben(input_coords):
    scaled_coords = input_coords * 10
    return scaled_coords


def normalize_coords(coords):
    coords = np.array(coords)
    max_c = np.max(coords[:, 0])
    min_c = np.min(coords[:, 0])
    coords[:, 0] = coords[:, 0] - min_c
    coords[:, 0] = coords[:, 0] / (max_c - min_c)
    max_c = np.max(coords[:, 1])
    min_c = np.min(coords[:, 1])
    coords[:, 1] = coords[:, 1] - min_c
    coords[:, 1] = coords[:, 1] / (max_c - min_c)
    return coords


def change_win_to_unix_path_if_needed(_str):
    if os.name == "unix" and _str.startswith("Y:"):
        _str = _str.replace("\\", "/").replace("Y:", "/cfs/share")
    if os.name == "unix" and _str.startswith("X:"):
        _str = _str.replace("\\", "/").replace("X:", "/cfs/home")
    return _str


if __name__ == '__main__':
    extract_nearest_mesh_nodes_to_sensors(
        Path(r'Y:\data\RTM\Leoben\sim_output\2019-07-23_15-38-08_5000p\0\2019-07-23_15-38-08_0'))
