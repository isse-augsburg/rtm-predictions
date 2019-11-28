import h5py
import numpy as np

import Pipeline.resampling as rs


def get_all_sensor_sequences(file, spacing=25, length=150):
    """
    Args: 
        file (string): File from which the data should be extracted.

    Returns: 
            [(data, label)]: all sequences of sensordata of a file as
            data and the filling percentage at the
            last sequence state as labels
    """
    sequences = []
    start = 0
    finish = length
    res = get_sensordata_and_filling_percentage(file, until=finish, frm=start)
    while res is not None:
        sequences.extend(res)
        start += spacing
        finish += spacing
        res = get_sensordata_and_filling_percentage(file, until=finish,
                                                    frm=start)

    if len(sequences) == 0:
        return None
    return sequences


def get_sensordata_and_filling_percentage(file, until=-1, frm=0):
    """
     Args: 
        file (string): File from which the data should be extracted.

    Returns: 
        [(data, label)]: sequence of sensordata as data and the filling
        percentage at the last sequence state as label
    """
    f = h5py.File(file, 'r')
    try:
        pressure_array = \
            f['post']['multistate']['TIMESERIES1']['multientityresults'][
                'SENSOR']['PRESSURE']['ZONE1_set1'][
                'erfblock'][
                'res'][()]
        all_states = f['post']['singlestate']
        _tmp = [state for state in all_states]
        last_filling = \
            f['post']['singlestate'][_tmp[-1]]['entityresults']['NODE'][
                'FILLING_FACTOR']['ZONE1_set1']['erfblock'][
                'res'][
                ()]
        non_zeros = np.count_nonzero(last_filling)
        state_count = np.shape(last_filling)[0]
        # filling = np.floor(non_zeros / state_count)
        filling = np.array([np.floor(non_zeros / state_count)])
        # filling_percentage = np.array((filling, 1 - filling), dtype=np.long)

    except KeyError:
        return None

    if np.shape(pressure_array)[0] < frm:
        return None
    # pressure_array = pressure_array[frm:until, :, :]
    pressure_array = pressure_array[frm:, :, :]
    pressure_array = pressure_array / 1000000
    # pressure_array = pressure_array[-frm:,:,:]
    pressure_array = np.squeeze(pressure_array)

    # print(np.shape(pressure_array), filling_percentage)

    # return ([(pressure_array, filling_percentage)])
    return [(pressure_array, filling)]


def sensorgrid_simulationsuccess(file, num_samples=50):
    data = get_sensordata_and_filling_percentage(file)

    if data is None:
        return None

    try:
        pressure_array, label = data[0]
        indices = rs.get_fixed_number_of_indices(len(pressure_array), num_samples)
        pressure_array = pressure_array[indices]
        pressure_array = np.where(pressure_array > 0, 1.0, 0.0)
        pressure_array = np.reshape(pressure_array, (38, 30, -1))
    except ValueError:
        print("File", file, "raised an error.")
        return None

    return [(pressure_array, label)]


def get_sensordata_and_filling_percentage_v2(file, until=400, frm=0):
    """
     Duplicate version of get_sensordata_and_fillingpercentage, should be
     deleted but might break something if so...."
    """
    f = h5py.File(file, 'r')
    try:
        pressure_array = \
            f['post']['multistate']['TIMESERIES1']['multientityresults'][
                'SENSOR']['PRESSURE']['ZONE1_set1'][
                'erfblock'][
                'res'][()]
        all_states = f['post']['singlestate']
        _tmp = [state for state in all_states]
        last_filling = \
            f['post']['singlestate'][_tmp[-1]]['entityresults']['NODE'][
                'FILLING_FACTOR']['ZONE1_set1']['erfblock'][
                'res'][
                ()]
        non_zeros = np.count_nonzero(last_filling)
        state_count = np.shape(last_filling)[0]
        filling_percentage = np.array(non_zeros / state_count)

    except KeyError:
        return None

    if np.shape(pressure_array)[0] < until:
        return None
    pressure_array = pressure_array[frm:until, :, :]
    pressure_array = np.squeeze(pressure_array)

    # print(np.shape(pressure_array), filling_percentage)

    return [(pressure_array, filling_percentage)]
