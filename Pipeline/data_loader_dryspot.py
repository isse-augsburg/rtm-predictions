import h5py
import numpy as np

from Utils.img_utils import normalize_coords, create_np_image


def get_flowfront_bool_dryspot_143x111(file, target_shape=(143, 111)):
    return get_flowfront_bool_dryspot(file, target_shape)


def get_flowfront_bool_dryspot(filename, target_shape, states=None, ignore_useless_states=False):
    """
    Load the flow front for the given states or all available states if states is None
    """
    f = h5py.File(filename, 'r')
    meta_file = h5py.File(str(filename).replace("RESULT.erfh5", "meta_data.hdf5"), 'r')
    try:
        if ignore_useless_states:
            useless_states = meta_file["useless_states/singlestates"][()]
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
            if ignore_useless_states and len(useless_states) > 0 and state == f'state{useless_states[0]:012d}':
                break
            label = 0
            if int(str(state).replace("state", "0")) in set_of_states:
                label = 1
            instances.append((create_np_image(target_shape=target_shape, norm_coords=_coords, data=filling), label))
        f.close()
        meta_file.close()
        return instances
    except KeyError:
        print(f'KeyError: {filename}')
        f.close()
        meta_file.close()
        return None


def get_sensor_bool_dryspot_ignore_useless_select_1_8(filename):
    return get_sensor_bool_dryspot(filename, True, selection_tuple=((1, 8), (1, 8)))


def get_sensor_bool_dryspot_ignore_useless(filename):
    return get_sensor_bool_dryspot(filename, True)


def get_sensor_bool_dryspot(filename, ignore_useless_states=False, selection_tuple=((0, 1), (0, 1))):
    """
    Load the flow front for the given states or all available states if states is None
    """
    f = h5py.File(filename, 'r')
    meta_file = h5py.File(str(filename).replace("RESULT.erfh5", "meta_data.hdf5"), 'r')
    try:
        array_of_states = meta_file["dryspot_states/singlestates"][()]
        if ignore_useless_states:
            useless_states = meta_file["useless_states/singlestates"][()]
        set_of_states = set(array_of_states.flatten())
        pressure_array = \
            f['post']['multistate']['TIMESERIES1']['multientityresults'][
                'SENSOR']['PRESSURE']['ZONE1_set1'][
                'erfblock'][
                'res'][()]
        instances = []
        states = f["post"]["singlestate"]
        for state in (states):
            if ignore_useless_states and len(useless_states) > 0 and state == f'state{useless_states[0]:012d}':
                break
            label = 0
            state_num = int(str(state).replace("state", "0"))
            if state_num in set_of_states:
                label = 1
            try:
                # Normalize data to fit betw. 0 and 1
                data = np.squeeze(pressure_array[state_num - 1]) / 100000
                if selection_tuple != ((0, 1), (0, 1)):
                    rect = data.reshape(38, 30)
                    sel = rect[selection_tuple[0][0]::selection_tuple[0][1],
                               selection_tuple[1][0]::selection_tuple[1][1]]
                    data = sel.flatten()
                instances.append((data, label))
            except IndexError:
                continue
        f.close()
        meta_file.close()
        return instances
    except KeyError:
        f.close()
        meta_file.close()
        return None


if __name__ == "__main__":
    things = get_sensor_bool_dryspot('/home/schroeter/Desktop/HDF5Dryspot/2019-07-23_15-38-08_0_RESULT.erfh5')
    # things = get_flowfront_bool_dryspot(r'X:\s\t\stiebesi\data\RTM\Leoben\output\with_shapes\2019-07-23_15-38'
    #                                     r'-08_5000p\0\2019-07-23_15-38-08_0_RESULT.erfh5', (143, 111))
    print('x')
