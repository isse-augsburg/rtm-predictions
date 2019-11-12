import logging

# import random
from functools import partial

import h5py
import numpy as np

# from Pipeline.data_gather import get_filelist_within_folder
# data_function must return [(data, label) ... (data, label)]
from plots_and_images import draw_polygon_map, plot_wrapper, scale_coords_leoben


# from PIL import Image

# FIXME: Lukas or Niklas, where is the load_image() function?
# def get_image_state_sequence(
#         folder, start_state=0, end_state=100, step=5, label_offset=3
# ):
#     filelist = get_filelist_within_folder(folder)
#     if len(filelist) == 0:
#         return None
#     filelist.sort()
#     state_list = list()
#
#     f = 0
#
#     c_state = start_state
#
#     label = None
#
#     while f < len(filelist):
#         f_meta = filelist[f]
#         f_split = f_meta.split("/")[-1].split("_")
#         state_index = int(f_split[0])
#
#         if state_index <= c_state:
#             state_list.append(load_image(f_meta))
#         f += 1
#         c_state += step
#
#         if c_state > end_state:
#             f += label_offset - 1
#             if f < len(filelist):
#                 f_meta = filelist[f]
#                 label = load_image(f_meta)
#
#             break
#
#     if label is None:
#         return None
#
#     data = np.stack(state_list)
#     if np.shape(data)[0] != int((end_state - start_state) / step) + 1:
#         return None
#
#     return [(data, label)]


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


def get_images_of_flow_front_and_permeability_map(filename, wanted_num=10, imsize=(155, 155)):
    f = h5py.File(filename, "r")
    im, scaled_coords, triangle_coords = get_local_properties_map(f, imsize)

    all_states = list(f["post"]["singlestate"].keys())
    selected_states = get_fixed_number_of_elements_and_indices(all_states, wanted_num=wanted_num)

    fillings = []
    for i, state in enumerate(all_states):
        try:
            filling_factor = f["post"]["singlestate"][state]["entityresults"]["NODE"]["FILLING_FACTOR"]["ZONE1_set1"][
                "erfblock"
            ]["res"][()]
        except KeyError:
            return None
        fillings.append(filling_factor)

    # indices = [int(x.split('state')[1]) for x in selected_states]
    indices = selected_states
    label = np.asarray(im)
    wrapper = partial(plot_wrapper, triangle_coords, scaled_coords, fillings, imsize)
    res = []
    for i in indices:
        try:
            res.append(wrapper(i))
        except IndexError or OSError:
            logger = logging.getLogger(__name__)
            logger.addHandler(logging.StreamHandler())
            logger.error(f"ERROR at {filename}, len(fillings): {len(fillings)}")
            # raise
    # array of all images, array of the same permeability map
    # trues, falses = 0, 0
    #
    # for i in images_and_indices:
    #     if i[2]:
    #         trues += 1
    #     else:
    #         falses += 1
    images = [x[0] for x in res]
    img_stack = np.stack(images)
    return [(img_stack[0:wanted_num], label)]


def get_fixed_number_of_elements_and_indices(input_list, wanted_num):
    num = len(input_list)
    dist = num / wanted_num
    if num == wanted_num:
        return input_list
    input_list.reverse()
    x = input_list[:: int(np.round(dist))]
    input_list.reverse()
    x.reverse()
    res = []
    for i in range(len(x)):
        res.append((len(input_list) - 1) - i * int(np.round(dist)))
    res.reverse()
    return res


def get_local_properties_map(f, imsize):
    coord_as_np_array = f["post/constant/entityresults/NODE/COORDINATE/ZONE1_set0/erfblock/res"][()]
    _all_coords = coord_as_np_array[:, :-1]

    scaled_coords = scale_coords_leoben(_all_coords)
    # norm_cords = normalize_coords(_all_coords)
    triangle_coords = f["post/constant/connectivities/SHELL/erfblock/ic"][()]
    triangle_coords = triangle_coords[:, :-1] - 151980  # required for Leoben data

    data = f["post/constant/entityresults/SHELL/"]

    im = create_local_properties_map(data, scaled_coords, triangle_coords, "FIBER_FRACTION")
    if im.size != imsize:
        im = im.resize(imsize)
    return im, scaled_coords, triangle_coords


def create_local_properties_map(data, scaled_coords, triangle_coords, _type="FIBER_FRACTION"):
    values_for_triangles = data[_type]["ZONE1_set1"]["erfblock"]["res"][()]
    im = draw_polygon_map(values_for_triangles, scaled_coords, triangle_coords, size=(152 * 3, 120 * 3))
    return im


def get_sensordata_and_flowfront_149x117(file, target_shape=(149, 117)):
    return get_sensordata_and_flowfront(file, target_shape)


def get_sensordata_and_flowfront_143x111(file, target_shape=(143, 111)):
    return get_sensordata_and_flowfront(file, target_shape)


def get_sensordata_and_flowfront_135x103(file, target_shape=(135, 103)):
    return get_sensordata_and_flowfront(file, target_shape)


def get_sensordata_and_flowfront_154x122(file, target_shape=(154, 122)):
    return get_sensordata_and_flowfront(file, target_shape)


def get_sensordata_and_flowfront(file, target_shape=(38, 30)):
    f = h5py.File(file, "r")
    instances = []
    try:
        coord_as_np_array = f["post/constant/entityresults/NODE/COORDINATE/ZONE1_set0/" "erfblock/res"][()]
        # Cut off last column (z), since it is filled with 1s anyway
        _coords = coord_as_np_array[:, :-1]

        _coords = normalize_coords(_coords)

        pressure_array = f["post"]["multistate"]["TIMESERIES1"]["multientityresults"]["SENSOR"]["PRESSURE"][
            "ZONE1_set1"
        ]["erfblock"]["res"][()]
        # convert barye to bar ( smaller values are more stable while training)
        pressure_array = pressure_array / 100000
        all_states = f["post"]["singlestate"]

        filling_factors_at_certain_times = [
            f["post"]["singlestate"][state]["entityresults"]["NODE"]["FILLING_FACTOR"]["ZONE1_set1"]["erfblock"]["res"][
                ()
            ]
            for state in all_states
        ]
        flat_fillings = np.squeeze(filling_factors_at_certain_times)
    except KeyError:
        return None

    for state, filling in zip(all_states, flat_fillings):
        try:
            s = state.replace("state", "")
            state_num = int(s)
            sensordata = np.squeeze(pressure_array[state_num - 1])
            arr = create_np_image(target_shape=target_shape, norm_coords=_coords, data=filling)
            instances.append((sensordata, arr))
        except IndexError:
            continue
    if len(instances) == 0:
        return None
    return instances


# FIXME also here, load image is used. where is it?
# def get_image_percentage(folder):
#     filelist = get_filelist_within_folder(folder)
#     if len(filelist) == 0:
#         return None
#     random.shuffle(filelist)
#
#     ret_list = []
#     for el in filelist:
#         f_split = el.split("/")[-1].split("_")
#         percentage = float(f_split[1])
#         dat = load_image(el)
#         ret_list.append((dat, np.array(percentage)))
#
#     return ret_list


if __name__ == "__main__":
    f = h5py.File("/home/schroeter/Desktop/2019-07-23_15-38-08_7_RESULT.erfh5", "r")
    im, scaled_coords, triangle_coords = get_local_properties_map(f, (152 * 3, 120 * 3))
    im.show()

    # get_sensordata_and_flowfront(
    #     Path(r"/home/schroeter/Desktop/2019-08-24_11-51-48_3_RESULT.erfh5")
    # )
