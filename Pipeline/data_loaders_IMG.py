import logging
from functools import partial

import h5py
import numpy as np

# from Pipeline.data_gather import get_filelist_within_folder
# data_function must return [(data, label) ... (data, label)]
from Pipeline.plots_and_images import draw_polygon_map, plot_wrapper
from Utils.img_utils import scale_coords_lautern, scale_coords_leoben, normalize_coords, create_np_image


def get_images_of_flow_front_and_permeability_map(filename, wanted_num=10, imsize=(155, 155)):
    f = h5py.File(filename, "r")
    im, scaled_coords, triangle_coords = get_local_properties_map_leoben(f, imsize)

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
            logger.error(f"ERROR at {filename}, len(fillings): {len(fillings)}")
            raise
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


# Deprecated! Use resampling.get_fixed_number_of_indices
def get_fixed_number_of_elements_and_indices(input_list, wanted_num):
    if wanted_num > len(input_list):
        return

    num = len(input_list)
    dist = num / wanted_num
    dist = int(np.floor(dist))
    if num == wanted_num:
        return input_list
    input_list.reverse()
    x = input_list[:: dist]
    input_list.reverse()
    x.reverse()
    res = []
    for i in range(len(x)):
        res.append((len(input_list) - 1) - i * dist)
    res.reverse()

    while len(res) > wanted_num:
        rnd_index = np.random.randint(0, len(res))
        res.pop(rnd_index)

    return res


def get_local_properties_map_lautern(f, imsize):
    coord_as_np_array = f["post/constant/entityresults/NODE/COORDINATE/ZONE1_set0/erfblock/res"][()]
    _all_coords = coord_as_np_array[:, :-1]
    scaled_coords = scale_coords_lautern(_all_coords)
    # norm_cords = normalize_coords(_all_coords)
    triangle_coords = f["post/constant/connectivities/SHELL/erfblock/ic"][()]
    triangle_coords = triangle_coords[:, :-1] - 1
    data = f["post/constant/entityresults/SHELL/"]

    im = create_local_properties_map(
        data, scaled_coords, triangle_coords, "FIBER_FRACTION"
    )
    if im.size != imsize:
        im = im.resize(imsize)
    return im, scaled_coords, triangle_coords


def get_local_properties_map_leoben(f, imsize):
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
    try:
        f = h5py.File(file, "r")
    except OSError:
        logger = logging.getLogger(__name__)
        logger.error(f"Error: File not found: {file}")
        return None

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


if __name__ == "__main__":
    # get_sensordata_and_flowfront(
    #     Path(r"/home/schroeter/Desktop/2019-08-24_11-51-48_3_RESULT.erfh5")
    # )
    f = h5py.File("/home/schroeter/Desktop/2019-07-23_15-38-08_7_RESULT.erfh5", "r")
    im, scaled_coords, triangle_coords = get_local_properties_map_lautern(f, (152 * 3, 120 * 3))
    im.show()
