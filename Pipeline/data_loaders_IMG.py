import io
import logging
import os
from pathlib import Path

import cv2
import h5py
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
from PIL import Image

# from Pipeline.data_gather import get_filelist_within_folder
# data_function must return [(data, label) ... (data, label)]
from Utils.img_utils import (
    normalize_coords,
    create_np_image,
)
from .resampling import get_fixed_number_of_indices


# This class provides all original functions but tries to improve the performance of consecutive calls
class DataloaderImages():
    def __init__(self, image_size=(135, 103)):
        self.image_size = image_size
        self.coords = None
        self.ff_coords = None
        self.fftriang = None

    def _get_flowfront(self, f, states=None):
        """
        Load the flow front for the given states or all available states if states is None
        """
        try:
            coords = self._get_coords(f)
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
            pressure_array = f["post"]["multistate"]["TIMESERIES1"][
                "multientityresults"
            ]["SENSOR"]["PRESSURE"]["ZONE1_set1"]["erfblock"]["res"][()]
            # convert barye to bar ( smaller values are more stable while training)
            pressure_array = pressure_array / 100000
            all_states = f["post"]["singlestate"]
        except KeyError:
            return None

        def sensordata_gen():
            for state in all_states:
                try:
                    s = state.replace("state", "")
                    state_num = int(s)
                    sensordata = np.squeeze(pressure_array[state_num - 1])
                    yield sensordata
                except IndexError:
                    yield None
        return sensordata_gen()

    def get_sensordata_and_flowfront(self, file):
        try:
            f = h5py.File(file, "r")
        except OSError:
            logger = logging.getLogger(__name__)
            logger.error(f"Error: File not found: {file}")
            return None

        fillings = self._get_flowfront(f)
        if not fillings:
            return None

        sensor_data = self._get_sensordata(f)
        if not sensor_data:
            return None

        # Return only tuples without None values and if we get no data at all, return None
        # `if not None in t` does not work here because numpy does some weird stuff on
        # such comparisons
        return (list(t for t in zip(sensor_data, fillings)
                     if t[0] is not None and t[1] is not None)
                or None)

    def _get_coords(self, f):
        if self.coords is not None:
            return self.coords
        coord_as_np_array = f[
            "post/constant/entityresults/NODE/COORDINATE/ZONE1_set0/erfblock/res"
        ][()]
        # Cut off last column (z), since it is filled with 1s anyway
        _coords = coord_as_np_array[:, :-1]

        _coords = normalize_coords(_coords)
        self.coords = _coords
        return _coords


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
    paths = [
        Path("/cfs/home/s/t/stiebesi/data/RTM/Leoben/output/with_shapes/2019-07-23_15-38-08_5000p/"
             "0/2019-07-23_15-38-08_0_RESULT.erfh5"),
        Path("/cfs/home/s/t/stiebesi/data/RTM/Leoben/output/with_shapes/2019-07-23_15-38-08_5000p/"
             "13/2019-07-23_15-38-08_13_RESULT.erfh5"),
    ]

    sh = logging.StreamHandler()
    sh.setLevel(logging.ERROR)
    logging.getLogger().addHandler(sh)
    thing = DataloaderImageSequences(wanted_frames=25)

    for path in paths:
        res = thing.get_sensordata_and_flowfront(path)
        res = thing.get_images_of_flow_front_and_permeability_map(
            path,
        )

        outdir = Path("/tmp/animout") / path.parts[-1] 
        os.makedirs(outdir, exist_ok=True)

        img_stack, label = res[0]
        img_stack *= 255
        img_stack = img_stack.astype(np.uint8)

        empty = np.zeros(label.shape, dtype=np.uint8)
        for i, frame in enumerate(img_stack):
            img_data = np.stack([frame, label, empty], axis=2)
            im = Image.fromarray(img_data, "RGB")
            im.save(outdir / "{:04}.png".format(i))

    # get_sensordata_and_flowfront(
    #     Path(r"/home/schroeter/Desktop/2019-08-24_11-51-48_3_RESULT.erfh5")
    # )
    # f = h5py.File("/home/schroeter/Desktop/2019-07-23_15-38-08_7_RESULT.erfh5", "r")
    # im, scaled_coords, triangle_coords = get_local_properties_map_lautern(f, (152 * 3, 120 * 3))
    # im.show()
    import pickle
    print(pickle.dumps(thing.get_images_of_flow_front_and_permeability_map))
