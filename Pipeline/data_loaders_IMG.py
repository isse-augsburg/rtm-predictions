import numpy as np
import h5py
import random
from PIL import Image
from Pipeline.data_gather import get_filelist_within_folder

def get_image_state_sequence(folder, start_state=0, end_state=100, step=5, label_offset=3):
    filelist = get_filelist_within_folder(folder)
    if len(filelist) == 0:
        return None
    filelist.sort()
    state_list = list()

    f = 0

    c_state = start_state

    label = None

    while f < len(filelist):
        f_meta = filelist[f]
        f_split = f_meta.split("/")[-1].split("_")
        state_index = int(f_split[0])

        if state_index <= c_state:
            state_list.append(load_image(f_meta))
        f += 1
        c_state += step

        if c_state > end_state:
            f += label_offset - 1
            if f < len(filelist):
                f_meta = filelist[f]
                label = load_image(f_meta)

            break

    if label is None:
        return None

    data = np.stack(state_list)
    if np.shape(data)[0] != int((end_state - start_state) / step) + 1:
        return None

    return [(data, label)]



def normalize_coords(coords):
    max_c = np.max(coords)
    min_c = np.min(coords)
    coords = coords - min_c
    coords = coords / (max_c - min_c)
    return coords


def create_np_image(target_shape=(151, 151), norm_coords=None, data=None, ):
    if norm_coords is None or data is None:
        print("ERROR")
        return
    assert np.shape(norm_coords)[0] == np.shape(data)[0]

    arr = np.zeros(target_shape)

    data = np.expand_dims(data, axis=1)
    coords_value = np.append(norm_coords, data, axis=1)
    coords_value[:, 0] = coords_value[:, 0] * (target_shape[0] - 1)
    coords_value[:, 1] = coords_value[:, 1] * (target_shape[1] - 1)
    coords_value[:, 2] = coords_value[:, 2]
    coords_value = coords_value.astype(np.int)
    arr[coords_value[:, 0], coords_value[:, 1]] = coords_value[:, 2]

    return arr


def get_sensordata_and_flowfront(file):
    f = h5py.File(file, 'r')
    instances = []
    try:

        coord_as_np_array = f['post/constant/entityresults/NODE/COORDINATE/ZONE1_set0/erfblock/res'][()]
        # Cut off last column (z), since it is filled with 1s anyway
        _coords = coord_as_np_array[:, :-1]
        _coords = normalize_coords(_coords)

        pressure_array = \
            f['post']['multistate']['TIMESERIES1']['multientityresults']['SENSOR']['PRESSURE']['ZONE1_set1'][
                'erfblock'][
                'res'][()]
        all_states = f['post']['singlestate']

        filling_factors_at_certain_times = [
            f['post']['singlestate'][state]['entityresults']['NODE']['FILLING_FACTOR']['ZONE1_set1']['erfblock']['res'][
                ()] for state in all_states]
        flat_fillings = np.squeeze(filling_factors_at_certain_times)
    except KeyError:
        return None
    for state, filling in zip(all_states, flat_fillings):
        try:
            s = state.replace("state", '')
            state_num = int(s)
            sensordata = np.squeeze(pressure_array[state_num - 1])
            # print(state_num, np.shape(filling), np.shape(sensordata))
            arr = create_np_image(norm_coords=_coords, data=filling)
            instances.append((sensordata, arr))
        except IndexError:
            continue
    if (len(instances) == 0):
        return None
    return instances


def get_image_percentage(folder):
    filelist = get_filelist_within_folder(folder)
    if len(filelist) == 0:
        return None
    random.shuffle(filelist)

    ret_list = []
    for el in filelist:
        f_split = el.split("/")[-1].split("_")
        percentage = float(f_split[1])
        dat = load_image(el)
        ret_list.append((dat, np.array(percentage)))

    return ret_list


def load_image(f_name):
    img = Image.open(f_name)
    img.load()
    data = np.asarray(img, dtype="float32")
    return data