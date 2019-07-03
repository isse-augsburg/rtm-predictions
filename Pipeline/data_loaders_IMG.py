import colorsys
import time
from functools import partial
from multiprocessing.pool import Pool
from pathlib import Path

import numpy as np
import h5py
import random
# from PIL import Image
from PIL import Image, ImageDraw


# from Pipeline.data_gather import get_filelist_within_folder

# data_function must return [(data, label) ... (data, label)]

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


def get_images_of_flow_front_and_permeability_map(filename, wanted_num=2, imsize=(155, 155), caching=True):
    cache_dir = ''
    if caching:
        # cache_dir = Path(f'/cfs/home/s/c/schroeni/Images/Cache/{filename.parts[-2]}/img_cache')
        cache_dir = filename.absolute().parent / 'img_cache'
        cache_dir.mkdir(parents=True, exist_ok=True)
    f = h5py.File(filename, 'r')
    im, scaled_coords, triangle_coords = get_local_properties_map(cache_dir, caching, f, imsize)

    all_states = list(f['post']['singlestate'].keys())
    selected_states = get_fixed_number_of_elements_and_their_indices_from_various_sized_list(all_states, wanted_num=wanted_num)

    fillings = []
    for i, state in enumerate(all_states):
        try:
            filling_factor = \
                f['post']['singlestate'][state]['entityresults']['NODE']['FILLING_FACTOR']['ZONE1_set1']['erfblock'][
                    'res'][()]
        except KeyError:
            return None
        fillings.append(filling_factor)

    indices = []
    for e in selected_states:
        indices.append(all_states.index(e))
    indices_and_their_names = zip(indices, selected_states)
    label = np.asarray(im)
    with Pool(4) as p:
        try:
            images_and_indices = p.map(
                partial(plot_wrapper, triangle_coords, scaled_coords, fillings, cache_dir, imsize),
                indices_and_their_names)
        except IndexError or OSError:
            print(f'ERROR at {filename}, len(fillings): {len(fillings)}')
            raise
    # array of all images, array of the same permeability map
    # trues, falses = 0, 0
    #
    # for i in images_and_indices:
    #     if i[2]:
    #         trues += 1
    #     else:
    #         falses += 1
    images = [x[0] for x in images_and_indices]
    img_stack = np.stack(images)
    return [(img_stack, label)]


def get_fixed_number_of_elements_and_their_indices_from_various_sized_list(input_list, wanted_num):
    num = len(input_list)
    if num == wanted_num:
        return input_list
    dist = num / wanted_num
    input_list.reverse()
    x = input_list[::int(np.round(dist))]
    input_list.reverse()
    x.reverse()
    return x


def get_local_properties_map(cache_dir, caching, f, imsize):
    coord_as_np_array = f['post/constant/entityresults/NODE/COORDINATE/ZONE1_set0/erfblock/res'][()]
    _all_coords = coord_as_np_array[:, :-1]
    scaled_coords = (_all_coords + 23.25) * 10
    # norm_cords = normalize_coords(_all_coords)
    triangle_coords = f['post/constant/connectivities/SHELL/erfblock/ic'][()]
    triangle_coords = triangle_coords[:, :-1] - 1
    data = f['post/constant/entityresults/SHELL/']
    if not (Path(cache_dir) / 'fiber_fraction.png').exists() and caching:
        im = create_local_properties_map(data, scaled_coords, triangle_coords, 'FIBER_FRACTION')
        im.save(Path(cache_dir) / 'fiber_fraction.png')
    else:
        im = Image.open(Path(cache_dir) / 'fiber_fraction.png')
    if im.size != imsize:
        im = im.resize(imsize)
    return im, scaled_coords, triangle_coords


def create_local_properties_map(data, scaled_coords, triangle_coords, _type='FIBER_FRACTION'):
    values_for_triangles = data[_type]['ZONE1_set1']['erfblock']['res'][()]
    im = draw_polygon_map(values_for_triangles, scaled_coords, triangle_coords)
    return im


def plot_wrapper(triangle_coords, scaled_coords, fillings, cache_dir, imsize, index_and_name):
    fillings = np.squeeze(fillings)
    index, index_name = index_and_name
    filling = fillings[index]
    im_fn = cache_dir / f'{index_name}.png'
    load_image = True
    if cache_dir != '' and not im_fn.exists():
        try:
            a = filling[triangle_coords]
            b = a.reshape(len(triangle_coords), 3)
            means_of_neighbour_nodes = b.mean(axis=1)
        except IndexError:
            print('ERROR plot wrapper ... raising')
            print(triangle_coords)
            print(filling[triangle_coords])
            raise

        im = draw_polygon_map(means_of_neighbour_nodes, scaled_coords, triangle_coords, colored=False)
        # im = create_np_image((465,465), scaled_coords, filling)
        # im_t = Image.fromarray(im,mode='L')
        im.save(im_fn)
        load_image = False
    else:
        try:
            im = Image.open(im_fn)
        except IndexError:
            print('ERROR: Corrupt img data')
            raise
    if im.size != imsize:
        im = im.resize(imsize)
    dat = np.asarray(im)
    im.close()
    return dat, index, load_image


def draw_polygon_map(values_for_triangles, scaled_coords, triangle_coords, colored=False, cache_dir=''):
    mode = 'RGB' if colored else 'L'
    im = Image.new(mode, (465, 465))
    draw = ImageDraw.Draw(im)
    for i in range(len(triangle_coords)):
        val = values_for_triangles[i]
        if not colored:
            draw.polygon(scaled_coords[triangle_coords[i]], fill=(int(val * 255)))
        else:
            if val == 0.0:
                draw.polygon(scaled_coords[triangle_coords[i]], fill=(255, 0, 0))
            elif val == 1.0:
                draw.polygon(scaled_coords[triangle_coords[i]], fill=(0, 102, 255))
            else:
                h = 3.6 * val
                col = tuple(int(round(i * 255)) for i in colorsys.hsv_to_rgb(h, 1, 1))
                draw.polygon(scaled_coords[triangle_coords[i]], fill=col)
    return im


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


if __name__ == "__main__":
    get_images_of_flow_front_and_permeability_map(Path(r'/home/niklas/Desktop/2019-05-17_16-45-57_0_RESULT.erfh5'))
