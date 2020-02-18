import logging

import h5py
import numpy as np

from Utils.data_utils import normalize_coords
from Utils.img_utils import create_np_image


class DataloaderDryspots:
    def __init__(self, image_size=None,
                 ignore_useless_states=True,
                 sensor_indizes=((0, 1), (0, 1)),
                 skip_indizes=(0, None, 1)):
        self.image_size = image_size
        self.ignore_useless_states = ignore_useless_states
        self.sensor_indizes = sensor_indizes
        self.skip_indizes = skip_indizes

    def get_flowfront_bool_dryspot(self, filename, states=None):
        """
        Load the flow front for the given states or all available states if states is None
        """
        f = h5py.File(filename, 'r')
        meta_file = h5py.File(str(filename).replace("RESULT.erfh5", "meta_data.hdf5"), 'r')
        try:
            if self.ignore_useless_states:
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

            states = list(states)[self.skip_indizes[0]:self.skip_indizes[1]:self.skip_indizes[2]]

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
                if self.ignore_useless_states and len(useless_states) > 0 and state == f'state{useless_states[0]:012d}':
                    break
                label = 0
                if int(str(state).replace("state", "0")) in set_of_states:
                    label = 1
                instances.append((create_np_image(target_shape=self.image_size, norm_coords=_coords, data=filling),
                                  label))
            f.close()
            meta_file.close()
            return instances
        except KeyError:
            logger = logging.getLogger()

            logger.warning(f'Warning: {filename}')
            f.close()
            meta_file.close()
            return None

    def get_sensor_bool_dryspot(self, filename):
        """
        Load the flow front for the given states or all available states if states is None
        """
        f = h5py.File(filename, 'r')
        meta_file = h5py.File(str(filename).replace("RESULT.erfh5", "meta_data.hdf5"), 'r')
        try:
            array_of_states = meta_file["dryspot_states/singlestates"][()]
            if self.ignore_useless_states:
                useless_states = meta_file["useless_states/singlestates"][()]
            set_of_states = set(array_of_states.flatten())
            pressure_array = \
                f['post']['multistate']['TIMESERIES1']['multientityresults'][
                    'SENSOR']['PRESSURE']['ZONE1_set1'][
                    'erfblock'][
                    'res'][()]
            instances = []
            states = f["post"]["singlestate"]

            states = list(states)[self.skip_indizes[0]:self.skip_indizes[1]:self.skip_indizes[2]]

            for state in states:
                if self.ignore_useless_states and len(useless_states) > 0 and state == f'state{useless_states[0]:012d}':
                    break
                label = 0
                state_num = int(str(state).replace("state", "0"))
                if state_num in set_of_states:
                    label = 1
                try:
                    # Normalize data to fit betw. 0 and 1
                    data = np.squeeze(pressure_array[state_num - 1]) / 100000
                    if self.sensor_indizes == ((0, 1), (0, 1)):
                        instances.append((data, label))
                    else:
                        rect = data.reshape(38, 30)
                        sel = rect[self.sensor_indizes[0][0]::self.sensor_indizes[0][1],
                                   self.sensor_indizes[1][0]::self.sensor_indizes[1][1]]
                        instances.append((sel.flatten(), label))
                except IndexError:
                    continue
            f.close()
            meta_file.close()
            return instances
        except KeyError:
            f.close()
            meta_file.close()
            return None
