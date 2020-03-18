import logging
import pickle
from collections import deque
from pathlib import Path

import h5py
import numpy as np

import Resources.training as r
from Utils.data_utils import extract_nearest_mesh_nodes_to_sensors, get_node_propery_at_states, \
    extract_coords_of_mesh_nodes
from Utils.img_utils import create_np_image


class DataloaderFlowfrontSensor:
    def __init__(self, image_size=None,
                 ignore_useless_states=True,
                 sensor_indizes=((0, 1), (0, 1)),
                 skip_indizes=(0, None, 1),
                 frame_count=1,
                 use_binary_sensor_only=False):
        self.image_size = image_size
        self.ignore_useless_states = ignore_useless_states
        self.sensor_indizes = sensor_indizes
        self.skip_indizes = skip_indizes
        self.mean = None
        self.std = None
        self.indeces_of_sensors = None
        self.frame_count = frame_count
        self.use_binary_sensor_only = use_binary_sensor_only
        self.coords = None
        self.load_ff_sensors()

    def load_ff_sensors(self):
        if r.nearest_nodes_to_sensors.is_file():
            with open(r.nearest_nodes_to_sensors, "rb") as nearest_nodes:
                _all_sensors = pickle.load(nearest_nodes)
        else:
            _all_sensors = extract_nearest_mesh_nodes_to_sensors(
                r.data_root / "2019-07-24_16-32-40_10p/0/2019-07-24_16-32-40_0")
            _all_sensors = _all_sensors.reshape((38, 30))
        with open(r.mean_std_20_flowfront_sensors, "rb") as mean_std:
            self.mean, self.std = pickle.load(mean_std)
        indices_of_sensors = _all_sensors[self.sensor_indizes[0][0]::self.sensor_indizes[0][1],
                                          self.sensor_indizes[1][0]::self.sensor_indizes[1][1]]
        self.indeces_of_sensors = indices_of_sensors.flatten()

    def extract_data_from_result_file(self, filename):
        with h5py.File(filename, 'r') as f:
            try:
                states = f["post"]["singlestate"]
                states = list(states)[self.skip_indizes[0]:self.skip_indizes[1]:self.skip_indizes[2]]

                filling_factors_at_certain_times = get_node_propery_at_states(f, "FILLING_FACTOR", states)
                velocity_at_certain_times = get_node_propery_at_states(f, "VELOCITY", states)
                fillings = np.squeeze(filling_factors_at_certain_times)
                velocities = np.squeeze(velocity_at_certain_times)
                self.extract_coords_data(f)
            except KeyError:
                logger = logging.getLogger()
                logger.warning(f'Warning: {filename}')
                return None, None, None
        return states, fillings, velocities

    def extract_data_from_meta_file(self, filename):
        with h5py.File(filename, 'r') as meta_file:
            try:
                useless_states = []
                if self.ignore_useless_states:
                    useless_states = meta_file["useless_states/singlestates"][()]
                array_of_states = meta_file["dryspot_states/singlestates"][()]
                set_of_dryspot_states = set(array_of_states.flatten())
            except KeyError:
                logger = logging.getLogger()
                logger.warning(f'Warning: {filename}')
                return None, None
        return useless_states, set_of_dryspot_states

    def extract_coords_data(self, f: h5py.File):
        if self.coords is not None:
            return self.coords
        self.coords = extract_coords_of_mesh_nodes(Path(f.filename))

    # TODO simplify code here, look at DataloaderImages for inspiration -> two generators (data, label) -> zip()
    def get_flowfront_sensor_and_flowfront_label(self, filename):
        """
        Load the flow front for the given states or all available states if states is None
        """
        states, fillings, velocities = self.extract_data_from_result_file(filename)
        meta_fn = str(filename).replace("RESULT.erfh5", "meta_data.hdf5")
        useless_states, set_of_dryspot_states = self.extract_data_from_meta_file(meta_fn)
        if states is None or \
                fillings is None or \
                velocities is None or \
                useless_states is None or \
                set_of_dryspot_states is None:
            return None
        instances = []
        frame_q = deque(maxlen=self.frame_count)
        for i, (velocity, filling, state) in enumerate(zip(velocities, fillings, states)):
            if self.ignore_useless_states \
                    and len(useless_states) > 0 \
                    and state == f'state{useless_states[0]:012d}':
                break
            binary_ff_sensor_values = np.ceil(filling[self.indeces_of_sensors])
            if self.use_binary_sensor_only:
                values = binary_ff_sensor_values
            else:
                values = self.add_velocity_factor(binary_ff_sensor_values, values, velocity)
            label = create_np_image(target_shape=self.image_size, norm_coords=self.coords, data=filling)
            if self.frame_count <= 1:
                instances.append((values, label))
            else:
                frame_q.append(values)
                """ 
                Stack the current frames of the queue so that each frame is in one channel and start 
                using the data just after having enough data in the queue
                """
                if self.frame_count > i + 1:
                    continue
                else:
                    instances.append((np.stack(list(frame_q), axis=1), label))

        return instances

    def get_flowfront_sensor_bool_dryspot(self, filename):
        """
        Load the flow front for the given states or all available states if states is None
        """
        states, fillings, velocities = self.extract_data_from_result_file(filename)
        meta_fn = str(filename).replace("RESULT.erfh5", "meta_data.hdf5")
        useless_states, set_of_dryspot_states = self.extract_data_from_meta_file(meta_fn)
        if states is None or \
                fillings is None or \
                velocities is None or \
                useless_states is None or \
                set_of_dryspot_states is None:
            return None
        instances = []
        frame_q = deque(maxlen=self.frame_count)
        for i, (velocity, filling, state) in enumerate(zip(velocities, fillings, states)):
            if self.ignore_useless_states \
                    and len(useless_states) > 0 \
                    and state == f'state{useless_states[0]:012d}':
                break
            binary_ff_sensor_values = np.ceil(filling[self.indeces_of_sensors])
            if self.use_binary_sensor_only:
                values = binary_ff_sensor_values
            else:
                values = self.add_velocity_factor(binary_ff_sensor_values, velocity)
            label = 0
            if int(str(state).replace("state", "0")) in set_of_dryspot_states:
                label = 1
            if self.frame_count <= 1:
                instances.append((values, label))
            else:
                frame_q.append(values)
                """ 
                Stack the current frames of the queue so that each frame is in one channel and start 
                using the data just after having enough data in the queue
                """
                if self.frame_count > i + 1:
                    continue
                else:
                    instances.append((np.stack(list(frame_q), axis=1), label))

        return instances

    def add_velocity_factor(self, binary_ff_sensor_values, velocity):
        velocities_at_sensors = velocity[self.indeces_of_sensors]
        abs_velocities_at_sensors = np.linalg.norm(velocities_at_sensors, axis=1)
        standardized_velocities = (abs_velocities_at_sensors - self.mean) / self.std
        return binary_ff_sensor_values * standardized_velocities


if __name__ == '__main__':
    dl = DataloaderFlowfrontSensor(sensor_indizes=((1, 8), (1, 8)))
    dl.get_flowfront_sensor_bool_dryspot(
        r'Y:\data\RTM\Leoben\sim_output\2019-07-24_16-32-40_10p\0\2019-07-24_16-32-40_0_RESULT.erfh5')
