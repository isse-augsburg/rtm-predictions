import logging
from functools import partial

import h5py
import numpy as np

# from Pipeline.data_gather import get_filelist_within_folder
# data_function must return [(data, label) ... (data, label)]
from Pipeline.plots_and_images import draw_polygon_map, plot_wrapper
from Utils.img_utils import (
    scale_coords_lautern,
    scale_coords_leoben,
    normalize_coords,
    create_np_image,
)


# This class provides all original functions but tries to improve the performance of consecutive calls
class DataloaderImages:
    def __init__(self, image_size=(155, 155)):
        self.image_size = image_size
        self.local_properties_leoben = None
        self.local_properties_lautern = None
        self.coords = None

    def get_images_of_flow_front_and_permeability_map(self, filename, wanted_num=10):
        f = h5py.File(filename, "r")
        im, scaled_coords, triangle_coords = self.__get_local_properties_map_leoben(
            f, self.image_size
        )

        all_states = list(f["post"]["singlestate"].keys())
        selected_states = self.__get_fixed_number_of_elements_and_indices(
            all_states, wanted_num=wanted_num
        )

        fillings = []
        for i, state in enumerate(all_states):
            try:
                filling_factor = f["post"]["singlestate"][state]["entityresults"][
                    "NODE"
                ]["FILLING_FACTOR"]["ZONE1_set1"]["erfblock"]["res"][()]
            except KeyError:
                return None
            fillings.append(filling_factor)

        # indices = [int(x.split('state')[1]) for x in selected_states]
        indices = selected_states
        label = np.asarray(im)
        wrapper = partial(
            plot_wrapper, triangle_coords, scaled_coords, fillings, self.image_size
        )
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
    def __get_fixed_number_of_elements_and_indices(self, input_list, wanted_num):
        if wanted_num > len(input_list):
            return

        num = len(input_list)
        dist = num / wanted_num
        dist = int(np.floor(dist))
        if num == wanted_num:
            return input_list
        input_list.reverse()
        x = input_list[::dist]
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

    def __get_local_properties_map_lautern(self, f, imsize):
        if self.local_properties_lautern is not None:
            return self.local_properties_lautern
        coord_as_np_array = f[
            "post/constant/entityresults/NODE/COORDINATE/ZONE1_set0/erfblock/res"
        ][()]
        _all_coords = coord_as_np_array[:, :-1]
        scaled_coords = scale_coords_lautern(_all_coords)
        # norm_cords = normalize_coords(_all_coords)
        triangle_coords = f["post/constant/connectivities/SHELL/erfblock/ic"][()]
        triangle_coords = triangle_coords[:, :-1] - 1
        data = f["post/constant/entityresults/SHELL/"]

        im = self.__create_local_properties_map(
            data, scaled_coords, triangle_coords, "FIBER_FRACTION"
        )
        if im.size != imsize:
            im = im.resize(imsize)
        self.local_properties_lautern = (im, scaled_coords, triangle_coords)
        return im, scaled_coords, triangle_coords

    def __get_local_properties_map_leoben(self, f, imsize):
        if self.local_properties_leoben is not None:
            return self.local_properties_leoben
        coord_as_np_array = f[
            "post/constant/entityresults/NODE/COORDINATE/ZONE1_set0/erfblock/res"
        ][()]
        _all_coords = coord_as_np_array[:, :-1]

        scaled_coords = scale_coords_leoben(_all_coords)
        # norm_cords = normalize_coords(_all_coords)
        triangle_coords = f["post/constant/connectivities/SHELL/erfblock/ic"][()]
        triangle_coords = triangle_coords[:, :-1] - 151980  # required for Leoben data

        data = f["post/constant/entityresults/SHELL/"]

        im = self.__create_local_properties_map(
            data, scaled_coords, triangle_coords, "FIBER_FRACTION"
        )
        if im.size != imsize:
            im = im.resize(imsize)
        self.local_properties_leoben = (im, scaled_coords, triangle_coords)
        return im, scaled_coords, triangle_coords

    def __create_local_properties_map(
        self, data, scaled_coords, triangle_coords, _type="FIBER_FRACTION"
    ):
        values_for_triangles = data[_type]["ZONE1_set1"]["erfblock"]["res"][()]
        im = draw_polygon_map(
            values_for_triangles,
            scaled_coords,
            triangle_coords,
            size=(152 * 3, 120 * 3),
        )
        return im

    def get_sensordata_and_flowfront(self, file):
        try:
            f = h5py.File(file, "r")
        except OSError:
            logger = logging.getLogger(__name__)
            logger.error(f"Error: File not found: {file}")
            return None

        instances = []
        try:
            _coords = self.__get_coords(f)

            pressure_array = f["post"]["multistate"]["TIMESERIES1"][
                "multientityresults"
            ]["SENSOR"]["PRESSURE"]["ZONE1_set1"]["erfblock"]["res"][()]
            # convert barye to bar ( smaller values are more stable while training)
            pressure_array = pressure_array / 100000
            all_states = f["post"]["singlestate"]

            filling_factors_at_certain_times = [
                f["post"]["singlestate"][state]["entityresults"]["NODE"][
                    "FILLING_FACTOR"
                ]["ZONE1_set1"]["erfblock"]["res"][()]
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
                arr = create_np_image(
                    target_shape=self.image_size, norm_coords=_coords, data=filling
                )
                instances.append((sensordata, arr))
            except IndexError:
                continue

        if len(instances) == 0:
            return None
        return instances

    def __get_coords(self, f):
        if self.coords is not None:
            return self.coords
        coord_as_np_array = f[
            "post/constant/entityresults/NODE/COORDINATE/ZONE1_set0/" "erfblock/res"
        ][()]
        # Cut off last column (z), since it is filled with 1s anyway
        _coords = coord_as_np_array[:, :-1]

        _coords = normalize_coords(_coords)
        self.coords = _coords
        return _coords


if __name__ == "__main__":
    # get_sensordata_and_flowfront(
    #     Path(r"/home/schroeter/Desktop/2019-08-24_11-51-48_3_RESULT.erfh5")
    # )
    # f = h5py.File("/home/schroeter/Desktop/2019-07-23_15-38-08_7_RESULT.erfh5", "r")
    # im, scaled_coords, triangle_coords = get_local_properties_map_lautern(f, (152 * 3, 120 * 3))
    # im.show()
    thing = DataloaderImages()
    import pickle
    print(pickle.dumps(thing.get_images_of_flow_front_and_permeability_map))
