import logging

import h5py
import numpy as np

from Utils.img_utils import normalize_coords, create_np_image


def get_flowfront_bool_dryspot_143x111(file, target_shape=(143, 111)):
    return get_flowfront_bool_dryspot(file, target_shape)


def get_flowfront_bool_dryspot(filename, target_shape, states=None):
    """
    Load the flow front for the given states or all available states if states is None
    """
    f = h5py.File(filename, 'r')
    meta_file = h5py.File(str(filename).replace("RESULT.erfh5", "meta_data.hdf5"), 'r')
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
            if int(str(state).replace("state", "0")) in set_of_states:
                label = 1
            instances.append((create_np_image(target_shape=target_shape, norm_coords=_coords, data=filling), label))
        f.close()
        meta_file.close()
        return instances
    except KeyError:
        logger = logging.getLogger(__name__)
        print(f'KeyError: {filename}')
        f.close()
        meta_file.close()
        return None


def get_sensor_bool_dryspot(filename):
    """
    Load the flow front for the given states or all available states if states is None
    """
    f = h5py.File(filename)
    meta_file = h5py.File(str(filename).replace("RESULT.erfh5", "meta_data.hdf5"))
    try:
        array_of_states = meta_file["dryspot_states/singlestates"][()]
        set_of_states = set(array_of_states.flatten())
        pressure_array = \
            f['post']['multistate']['TIMESERIES1']['multientityresults'][
                'SENSOR']['PRESSURE']['ZONE1_set1'][
                'erfblock'][
                'res'][()]
        instances = []
        states = f["post"]["singlestate"]
        for state in (states):
            label = 0
            state_num = int(str(state).replace("state", "0"))
            if (state_num in set_of_states):
                label = 1
            try:
                instances.append((np.squeeze(pressure_array[state_num - 1]) / 100000, label))
            except IndexError:
                continue

        return instances
    except KeyError:
        return None


if __name__ == "__main__":
    things = get_sensor_bool_dryspot('/home/schroeter/Desktop/HDF5Dryspot/2019-07-23_15-38-08_0_RESULT.erfh5')
    # things = get_flowfront_bool_dryspot(r'X:\s\t\stiebesi\data\RTM\Leoben\output\with_shapes\2019-07-23_15-38'
    #                                     r'-08_5000p\0\2019-07-23_15-38-08_0_RESULT.erfh5', (143, 111))
    print('x')
