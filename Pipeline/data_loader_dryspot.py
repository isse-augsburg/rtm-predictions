import logging

import h5py
import numpy as np


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
    # for new data 38 and 30.0


def create_np_image(target_shape=(143, 111), norm_coords=None, data=None):
    if norm_coords is None or data is None:
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.StreamHandler())
        logger.error("ERROR in create_np_image")
        return

    assert np.shape(norm_coords)[0] == np.shape(data)[0]

    arr = np.zeros(target_shape)

    data = np.expand_dims(data, axis=1)
    coords_value = np.append(norm_coords, data, axis=1)
    coords_value[:, 0] = coords_value[:, 0] * (target_shape[0] - 1)
    coords_value[:, 1] = coords_value[:, 1] * (target_shape[1] - 1)
    coords_value[:, 2] = coords_value[:, 2]
    # coords_value = coords_value.astype(np.int)
    arr[coords_value[:, 0].astype(np.int), coords_value[:, 1].astype(np.int)] = coords_value[:, 2]

    return arr


def get_flowfront_bool_dryspot(filename, target_shape, states=None):
    """
    Load the flow front for the given states or all available states if states is None
    """
    f = h5py.File(filename)
    meta_file = h5py.File(str(filename).replace("RESULT.erfh5", "meta_data.hdf5"))
    try:
        array_of_states = meta_file["dryspot_states/singlestates"][()]
        set_of_states = set(array_of_states.flatten())
        coord_as_np_array = f[
            "post/constant/entityresults/NODE/COORDINATE/ZONE1_set0/" 
            "erfblock/res"
        ][()]
        # Cut off last column (z), since it is filled with 1s anyway
        _coords = coord_as_np_array[:, :-1]
        _coords = normalize_coords(_coords)
        if not states:
            states = f["post"]["singlestate"]

        filling_factors_at_certain_times = [
            f["post"]["singlestate"][state]["entityresults"]["NODE"][
                "FILLING_FACTOR"][
                "ZONE1_set1"
            ]["erfblock"]["res"][()]
            for state in states
        ]

        flat_fillings = np.squeeze(filling_factors_at_certain_times)
        instances = []
        for filling, state in zip(flat_fillings, states):
            label = 0
            if (int(str(state).replace("state", "0")) in set_of_states):
                label = 1
            instances.append((create_np_image(target_shape=target_shape, norm_coords=_coords, data=filling), label))
        return instances
    except KeyError:
        return None


if __name__ == "__main__":
    # things = get_flowfront_bool_dryspot('/home/schroeter/Desktop/HDF5Dryspot/2019-07-23_15-38-08_0_RESULT.erfh5',
    # (143, 111))
    things = get_flowfront_bool_dryspot(r'X:\s\t\stiebesi\data\RTM\Leoben\output\with_shapes\2019-07-23_15-38'
                                        r'-08_5000p\0\2019-07-23_15-38-08_0_RESULT.erfh5', (143, 111))
    print('x')
