import numpy as np
import h5py


def get_all_sensor_sequences(file, spacing=25, length=150):
    l = []
    start = 0
    finish = length
    res = get_sensordata_and_filling_percentage(file, until=finish, frm=start)
    while res is not None:
        l.extend(res)
        start += spacing
        finish += spacing
        res = get_sensordata_and_filling_percentage(file, until=finish, frm=start)

    if len(l) == 0:
        return None
    return l


def get_sensordata_and_filling_percentage(file, until=-1, frm=0):
    f = h5py.File(file, 'r')
    try:
        pressure_array = \
            f['post']['multistate']['TIMESERIES1']['multientityresults']['SENSOR']['PRESSURE']['ZONE1_set1'][
                'erfblock'][
                'res'][()]
        all_states = f['post']['singlestate']
        _tmp = [state for state in all_states]
        last_filling = \
            f['post']['singlestate'][_tmp[-1]]['entityresults']['NODE']['FILLING_FACTOR']['ZONE1_set1']['erfblock'][
                'res'][
                ()]
        non_zeros = np.count_nonzero(last_filling)
        state_count = np.shape(last_filling)[0]
        filling = np.floor(non_zeros / state_count)
        filling_percentage = np.array((filling, 1 - filling), dtype=np.long)

    except KeyError:
        return None

    if (np.shape(pressure_array)[0] < frm):
        return None
    pressure_array = pressure_array[frm:until, :, :]
    # pressure_array = pressure_array[-frm:,:,:]
    pressure_array = np.squeeze(pressure_array)

    # print(np.shape(pressure_array), filling_percentage)

    return ([(pressure_array, filling_percentage)])


def get_sensordata_and_filling_percentage_v2(file, until=400, frm=0):
    f = h5py.File(file, 'r')
    try:
        pressure_array = \
            f['post']['multistate']['TIMESERIES1']['multientityresults']['SENSOR']['PRESSURE']['ZONE1_set1'][
                'erfblock'][
                'res'][()]
        all_states = f['post']['singlestate']
        _tmp = [state for state in all_states]
        last_filling = \
            f['post']['singlestate'][_tmp[-1]]['entityresults']['NODE']['FILLING_FACTOR']['ZONE1_set1']['erfblock'][
                'res'][
                ()]
        non_zeros = np.count_nonzero(last_filling)
        state_count = np.shape(last_filling)[0]
        filling_percentage = np.array(non_zeros / state_count)

    except KeyError:
        return None

    if (np.shape(pressure_array)[0] < until):
        return None
    pressure_array = pressure_array[frm:until, :, :]
    pressure_array = np.squeeze(pressure_array)

    # print(np.shape(pressure_array), filling_percentage)

    return ([(pressure_array, filling_percentage)])
