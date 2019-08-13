import h5py
import matplotlib.pyplot as plt
import numpy as np
from os import walk
from tqdm import tqdm

def parse_out_file(filename):
    f = open(filename, "r")
    lines = f.readlines()

    all_sensors = []

    id_string = "sensor number:"
    coords_string = "sensor location:"
    sensor_index, sensor_coords = 0, (0,0)
    
    for line in lines:
        line = line.strip()
        
        if id_string in line:
            slice = line.split(":")[-1]
            sensor_index = int(slice) - 1
            

        if coords_string in line: 
            slice = line.split(":")[-1].strip()
            slice = slice.split(" ")[:-1]
            slice = [float(x) for x in slice]
            sensor_coords = tuple(slice)
            all_sensors.append([sensor_index, sensor_coords])
            sensor_index, sensor_coords = 0, (0,0)
    
    return all_sensors


def get_all_sensor_values(filename):
    f = h5py.File(filename, 'r')

    pressure_array = f['post']['multistate']['TIMESERIES1']['multientityresults']['SENSOR']['PRESSURE']['ZONE1_set1']['erfblock']['res'][()]
    pressure_array = pressure_array / 100 

    return pressure_array

def plot_sensorgrid(filename_data, filename_sensorinfo):
    all_sensors = parse_out_file(filename_sensorinfo)
    sensor_values = get_all_sensor_values(filename_data)

    for i, v in enumerate(sensor_values): 
        if i % 50 == 0: 
            for sensor_id,(x, y) in all_sensors:
                value = int(v[sensor_id])
                #color = np.array([[min(value, 255), 255, 255]])

                color = np.array([[min(max(255-value, 0), 255), 255, 255]])
                plt.scatter(x, y, c=color/255)

            plt.show()


if __name__ == "__main__":
    filename_sensorinfo = '/run/user/1001/gvfs/smb-share:server=137.250.170.56,share=share/data/RTM/Leoben/output/with_shapes/2019-07-23_15-38-08_5000p/39/2019-07-23_15-38-08_39d.out'
    filename_data = '/run/user/1001/gvfs/smb-share:server=137.250.170.56,share=share/data/RTM/Leoben/output/with_shapes/2019-07-23_15-38-08_5000p/39/2019-07-23_15-38-08_39_RESULT.erfh5'
    #parse_out_file(filename)
    plot_sensorgrid(filename_data, filename_sensorinfo)