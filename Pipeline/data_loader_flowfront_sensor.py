import logging
from pathlib import Path

import h5py
import numpy as np

from Utils.data_utils import extract_nearest_mesh_nodes_to_sensors


class DataloaderFlowfrontSensor:
    def __init__(self, image_size=None,
                 ignore_useless_states=True,
                 sensor_indizes=((0, 1), (0, 1)),
                 skip_indizes=(0, None, 1)):
        self.image_size = image_size
        self.ignore_useless_states = ignore_useless_states
        self.sensor_indizes = sensor_indizes
        self.skip_indizes = skip_indizes
        _all_sensors = extract_nearest_mesh_nodes_to_sensors(
            Path(r'Y:\data\RTM\Leoben\sim_output\2019-07-24_16-32-40_10p\0\2019-07-24_16-32-40_0'))
        _all_sensors = _all_sensors.reshape((38, 30))
        indices_of_sensors = _all_sensors[sensor_indizes[0][0]::sensor_indizes[0][1],
                                          sensor_indizes[1][0]::sensor_indizes[1][1]]
        self.indeces_of_sensors = indices_of_sensors.flatten()

    def get_flowfront_sensor_bool_dryspot(self, filename):
        """
        Load the flow front for the given states or all available states if states is None
        """
        f = h5py.File(filename, 'r')
        meta_file = h5py.File(str(filename).replace("RESULT.erfh5", "meta_data.hdf5"), 'r')
        try:
            useless_states = []
            if self.ignore_useless_states:
                useless_states = meta_file["useless_states/singlestates"][()]
            array_of_states = meta_file["dryspot_states/singlestates"][()]
            set_of_states = set(array_of_states.flatten())
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
                sensor_values = np.ceil(filling[self.indeces_of_sensors])

                if self.ignore_useless_states and len(useless_states) > 0 and state == f'state{useless_states[0]:012d}':
                    break
                label = 0
                if int(str(state).replace("state", "0")) in set_of_states:
                    label = 1
                instances.append((sensor_values, label))
            f.close()
            meta_file.close()
            return instances
        except KeyError:
            logger = logging.getLogger()

            logger.warning(f'Warning: {filename}')
            f.close()
            meta_file.close()
            return None


if __name__ == '__main__':
    dl = DataloaderFlowfrontSensor(sensor_indizes=((1, 8), (1, 8)))
    dl.get_flowfront_sensor_bool_dryspot(
        r'Y:\data\RTM\Leoben\sim_output\2019-07-24_16-32-40_10p\0\2019-07-24_16-32-40_0_RESULT.erfh5')
