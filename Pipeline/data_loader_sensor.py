import h5py
import numpy as np

import Pipeline.resampling as rs


class DataLoaderSensor:
    def __init__(self,
                 num_samples_sensorgrid=50,
                 frm=0,
                 until=-1):

        self.num_samples = num_samples_sensorgrid
        self.frm = frm
        self.until = until

    def get_sensordata_and_filling_percentage(self, file):
        """
         Args:
            file (string): File from which the data should be extracted.
            until: End frame of sequence.
            frm: Starting frame of sequence.

        Returns:
            [(data, label)]: sequence of sensordata as data and the filling
            percentage at the last sequence state as label
        """
        f = h5py.File(file, 'r')
        try:
            pressure_array = \
                f['post']['multistate']['TIMESERIES1']['multientityresults'][
                    'SENSOR']['PRESSURE']['ZONE1_set1'][
                    'erfblock'][
                    'res'][()]
            all_states = f['post']['singlestate']
            _tmp = [state for state in all_states]
            last_filling = \
                f['post']['singlestate'][_tmp[-1]]['entityresults']['NODE'][
                    'FILLING_FACTOR']['ZONE1_set1']['erfblock'][
                    'res'][
                    ()]
            non_zeros = np.count_nonzero(last_filling)
            state_count = np.shape(last_filling)[0]
            filling = np.array([np.floor(non_zeros / state_count)])

        except KeyError:
            return None

        if np.shape(pressure_array)[0] < self.frm:
            return None
        pressure_array = pressure_array[self.frm:, :, :]
        pressure_array = pressure_array / 1000000
        pressure_array = np.squeeze(pressure_array)

        return [(pressure_array, filling)]

    def sensorgrid_simulationsuccess(self, file):
        data = self.get_sensordata_and_filling_percentage(file)

        if data is None:
            return None

        try:
            pressure_array, label = data[0]
            indices = rs.get_fixed_number_of_indices(len(pressure_array), self.num_samples)
            pressure_array = pressure_array[indices]
            pressure_array = np.where(pressure_array > 0, 1.0, 0.0)
            pressure_array = np.reshape(pressure_array, (38, 30, -1))
        except ValueError:
            print("File", file, "raised an error.")
            return None

        return [(pressure_array, label)]
