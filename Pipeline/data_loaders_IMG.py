import io
import logging
from pathlib import Path

import cv2
import h5py
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np

import Resources.training as tr_resources
from Pipeline.resampling import get_fixed_number_of_indices
from Utils.data_utils import extract_coords_of_mesh_nodes, load_mean_std
# from Pipeline.data_gather import get_filelist_within_folder
# data_function must return [(data, label) ... (data, label)]
from Utils.img_utils import (
    create_np_image,
)
from Utils.img_utils import flip_array_diag


# This class provides all original functions but tries to improve the performance of consecutive calls
class DataloaderImages:
    def __init__(self, image_size=(135, 103),
                 ignore_useless_states=True,
                 sensor_indizes=((0, 1), (0, 1)),
                 skip_indizes=(0, None, 1),
                 divide_by_100k=True):
        self.image_size = image_size
        self.coords = None
        self.ff_coords = None
        self.fftriang = None
        self.ignore_useless_states = ignore_useless_states
        self.sensor_indizes = sensor_indizes
        self.skip_indizes = skip_indizes
        self.divide_by_100k = divide_by_100k
        self.mean = None
        self.std = None
        if not self.divide_by_100k:
            self.mean, self.std = load_mean_std(tr_resources.mean_std_1140_pressure_sensors)

    def _get_flowfront(self, f: h5py.File, meta_f: h5py.File, states=None):
        """
        Load the flow front for the given states or all available states if states is None
        """
        useless_states = None
        try:
            coords = self._get_coords(f)
            if not states:
                states = f["post"]["singlestate"]
            states = list(states)[self.skip_indizes[0]:self.skip_indizes[1]:self.skip_indizes[2]]
            if meta_f is not None:
                useless_states = meta_f["useless_states/singlestates"][()]
                if len(useless_states) == 0:
                    useless_states = None
            filling_factors_at_certain_times = []
            for state in states:
                if useless_states is not None and state == f'state{useless_states[0]:012d}':
                    break
                else:
                    filling_factors_at_certain_times.append(f["post"][
                                                            "singlestate"][state]["entityresults"]["NODE"][
                                                            "FILLING_FACTOR"][
                                                            "ZONE1_set1"][
                                                            "erfblock"]["res"][()])

            flat_fillings = np.squeeze(filling_factors_at_certain_times)
            return (create_np_image(target_shape=self.image_size, norm_coords=coords, data=filling)
                    for filling in flat_fillings)
        except KeyError:
            return None

    def _get_fiber_fraction(self, f):
        if self.ff_coords is None:
            coords = self._get_coords(f).copy()
            x = coords[:, 0]
            y = coords[:, 1]
            x *= 375
            y *= 300
            self.ff_coords = x, y
        x, y = self.ff_coords

        if self.fftriang is None:
            triangles = f["/post/constant/connectivities/SHELL/erfblock/ic"][()]
            triangles = triangles - triangles.min()
            triangles = triangles[:, :-1]
            xi = np.linspace(0, 375, 376)
            yi = np.linspace(0, 300, 301)
            Xi, Yi = np.meshgrid(xi, yi)
            self.fftriang = tri.Triangulation(x, y, triangles=triangles)

        # Fiber fraction map creation with tripcolor
        fvc = f["/post/constant/entityresults/SHELL/FIBER_FRACTION/ZONE1_set1/erfblock/res"][()].flatten()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.tripcolor(self.fftriang, fvc, cmap="gray")

        ax.set(xlim=(0, 375), ylim=(0, 300))
        ax.axis("off")
        fig.set_tight_layout(True)

        extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        for im in ax.get_images():
            im.set_clim(0, 1)

        perm_bytes = io.BytesIO()
        fig.savefig(perm_bytes, bbox_inches=extent, format="png")
        plt.close(fig)
        perm_bytes.seek(0)

        file_bytes = np.asarray(perm_bytes.getbuffer(), dtype=np.uint8)
        perm_map = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        perm_map = cv2.resize(perm_map, self.image_size)
        perm_map = cv2.rotate(perm_map, cv2.ROTATE_90_CLOCKWISE)

        return perm_map

    def _get_sensordata(self, f):
        try:
            data = f["post"]["multistate"]["TIMESERIES1"][
                "multientityresults"
            ]["SENSOR"]["PRESSURE"]["ZONE1_set1"]["erfblock"]["res"][()]

            states = f["post"]["singlestate"]
        except KeyError:
            return None

        states = list(states)[self.skip_indizes[0]:self.skip_indizes[1]:self.skip_indizes[2]]

        def sensordata_gen():
            for state in states:
                try:
                    s = state.replace("state", "")
                    state_num = int(s)
                    sensordata = np.squeeze(data[state_num - 1])
                    if self.divide_by_100k:
                        # convert barye to bar ( smaller values are more stable while training)
                        sensordata = sensordata / 100000
                    else:
                        # Standardize
                        sensordata = (sensordata - self.mean) / self.std
                    if self.sensor_indizes != ((0, 1), (0, 1)):
                        sensordata = sensordata.reshape((38, 30))
                        sensordata = sensordata[self.sensor_indizes[0][0]::self.sensor_indizes[0][1],
                                                self.sensor_indizes[1][0]::self.sensor_indizes[1][1]]
                        sensordata = sensordata.flatten()
                    yield sensordata
                except IndexError:
                    yield None

        return sensordata_gen()

    def get_sensordata_and_flowfront(self, file: Path):
        try:
            result_f = h5py.File(file, "r")
            if self.ignore_useless_states:
                meta_f = h5py.File(str(file).replace("RESULT.erfh5", "meta_data.hdf5"), 'r')
            else:
                meta_f = None
        except OSError:
            logger = logging.getLogger(__name__)
            logger.error(f"Error: File not found: {file} (or meta_data.hdf5)")
            return None

        fillings = self._get_flowfront(result_f, meta_f)
        if not fillings:
            return None

        sensor_data = self._get_sensordata(result_f)
        if not sensor_data:
            return None

        # Return only tuples without None values and if we get no data at all, return None
        # `if not None in t` does not work here because numpy does some weird stuff on
        # such comparisons
        return (list((sens_data, filling, {"state": state}) for sens_data, filling, state in
                     zip(sensor_data, fillings, result_f["post"]["singlestate"])
                     if sens_data is not None and filling is not None)
                or None)

    def _get_coords(self, f: h5py.File):
        if self.coords is not None:
            return self.coords
        self.coords = extract_coords_of_mesh_nodes(Path(f.filename))
        return self.coords


class DataloaderImageSequences(DataloaderImages):
    """
    Subclass for dataloader functions that generate sequences of images
    as samples and only generate one sample per file
    """

    def __init__(self, image_size=(135, 103), wanted_frames=10):
        super().__init__(image_size=image_size)
        self.wanted_frames = wanted_frames

    def get_images_of_flow_front_and_permeability_map(self, filename):
        logger = logging.getLogger(__name__)
        logger.debug("Loading flow front and premeability maps from {}".format(filename))
        f = h5py.File(filename, "r")

        perm_map = self._get_fiber_fraction(f)
        perm_map = perm_map.astype(np.float) / 255

        all_states = list(f["post"]["singlestate"].keys())
        indices = get_fixed_number_of_indices(len(all_states), self.wanted_frames)
        if indices is None:
            return None
        try:
            wanted_states = [all_states[i] for i in indices]
        except IndexError or OSError:
            logger.error(f"ERROR at {filename}, available states: {all_states},"
                         f"wanted indices: {indices}")
            raise
        ffgen = self._get_flowfront(f, states=wanted_states)
        if ffgen is None:
            return None
        images = list(ffgen)

        img_stack = np.stack(images)
        return [(img_stack[0:self.wanted_frames], perm_map)]


if __name__ == "__main__":
    dl = DataloaderImages()

    paths = [
        Path("/cfs/home/s/t/stiebesi/data/RTM/Leoben/output/with_shapes/2019-07-23_15-38-08_5000p/"
             "40/2019-07-23_15-38-08_40_RESULT.erfh5"),

    ]
    for p in paths:
        res = dl.get_sensordata_and_flowfront(p)
        for thing in res:
            plt.imshow(thing[1])
            plt.imshow(flip_array_diag(thing[1]))
