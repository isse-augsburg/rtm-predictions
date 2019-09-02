import glob
import os
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import regex as re
from prettytable import PrettyTable


class HDF5Object:
    def __init__(self, meta_path, result_path):
        self.output_frequency_type_path =       "output_frequency_type"
        # Pertubations_Factors
        self.general_sigma_path =               "perturbation_factors/General_Sigma"
        # -Shapes
        self.number_of_circles_path =           "perturbation_factors/Shapes/Circles/Num"
        self.number_of_rectangles_path =        "perturbation_factors/Shapes/Rectangles/Num"
        self.number_of_runners_path =           "perturbation_factors/Shapes/Runners/Num"
        self.fibre_content_circles_path =       "perturbation_factors/Shapes/Circles/Fiber_Content"
        self.fibre_content_rectangles_path =    "perturbation_factors/Shapes/Rectangles/Fiber_Content"
        self.fibre_content_runners_path =       "perturbation_factors/Shapes/Runners/Fiber_Content"

        # Shapes
        # -Circlepaths
        self.fvc_circle_path      = "shapes/Circle/fvc"
        self.radius_circle_path   = "shapes/Circle/radius"
        self.posx_circle_path     = "shapes/Circle/posX"
        self.posy_circle_path     = "shapes/Circle/posY"
        # -Rectanglepaths
        self.fvc_rectangle_path =       "shapes/Rectangle/fvc"
        self.height_rectangle_path =    "shapes/Rectangle/height"
        self.width_rectangle_path =     "shapes/Rectangle/width"
        self.posx_rectangle_path =      "shapes/Rectangle/posX"
        self.posy_rectangle_path =      "shapes/Rectangle/posY"
        # -Runnerspaths
        self.fvc_runner_path =             "shapes/Runner/fvc"
        self.height_runner_path =          "shapes/Runner/height"
        self.width_runner_path =           "shapes/Runner/width"
        self.posx_runner_path =            "shapes/Runner/posX"
        self.posy_runner_path =            "shapes/Runner/posY"
        self.pos_lower_leftx_runner_path = "shapes/Runner/posLowerLeftX"
        self.pos_lower_lefty_runner_path = "shapes/Runner/posLowerLeftY"

        self.single_state_path =        "post/singlestate"
        self.number_of_sensors_path =   "post/multistate/TIMESERIES1/multientityresults/SENSOR/PRESSURE/ZONE1_set1/erfblock/res"

        self.number_of_circles = 0
        self.number_of_rectangles = 0
        self.number_of_runners = 0
        self.number_of_sensors = 0

        # Metadata
        m = h5py.File(meta_path, 'r')
        self.meta_path = meta_path
        if(self.output_frequency_type_path in m):
            self.output_frequency_type = m[self.output_frequency_type_path][()]

        if(self.general_sigma_path in m):
            self.general_sigma = m[self.general_sigma_path][()]
        if(self.number_of_circles_path in m):
            self.number_of_circles = m[self.number_of_circles_path][()]
        if(self.number_of_rectangles_path in m):
            self.number_of_rectangles = m[self.number_of_rectangles_path][()]
        if(self.number_of_runners_path in m):
            self.number_of_runners = m[self.number_of_runners_path][()]
        self.number_of_shapes = self.number_of_circles + self.number_of_rectangles + self.number_of_runners
        if(self.fibre_content_circles_path in m):
            self.fibre_content_circles = m[self.fibre_content_circles_path][()]
        if(self.fibre_content_rectangles_path in m):
            self.fibre_content_rectangles = m[self.fibre_content_rectangles_path][()]

        if(self.fibre_content_runners_path in m):
            self.fibre_content_runners = m[self.fibre_content_runners_path][()]

        # Shapes
        # -Circle
        if(self.fvc_circle_path in m):
            self.fvc_circle = np.array(m[self.fvc_circle_path][()])
        if(self.radius_circle_path in m):
            self.radius_circle = np.array(m[self.radius_circle_path][()])
        if(self.posx_circle_path in m):
            self.posx_circle = np.array(m[self.posx_circle_path][()])
        if(self.posy_circle_path in m):
            self.posy_circle = np.array(m[self.posy_circle_path][()])
        # -Rectangle
        if(self.fvc_rectangle_path in m):
            self.fvc_rectangle = np.array(m[self.fvc_rectangle_path][()])
        if(self.height_rectangle_path in m):
            self.height_rectangle = np.array(m[self.height_rectangle_path][()])
        if(self.width_rectangle_path in m):
            self.width_rectangle = np.array(m[self.width_rectangle_path][()])
        if(self.posx_rectangle_path in m):
            self.posx_rectangle = np.array(m[self.posx_rectangle_path][()])
        if(self.posy_rectangle_path in m):
            self.posy_rectangle = np.array(m[self.posy_rectangle_path][()])
        # -Runner
        if(self.fvc_runner_path in m):
            self.fvc_runner = np.array(m[self.fvc_runner_path][()])
        if(self.height_runner_path in m):
            self.height_runner = np.array(m[self.height_runner_path][()])
        if(self.width_runner_path in m):
            self.width_runner = np.array(m[self.width_runner_path][()])
        if(self.posx_runner_path in m):
            self.posx_runner = np.array(m[self.posx_runner_path][()])
        if(self.posy_runner_path in m):
            self.posy_runner = np.array(m[self.posy_runner_path][()])
        if(self.pos_lower_leftx_runner_path in m):
            self.pos_lower_leftx_runner = np.array(m[self.pos_lower_leftx_runner_path][()])
        if(self.pos_lower_lefty_runner_path in m):
            self.pos_lower_lefty_runner = np.array(m[self.pos_lower_lefty_runner_path][()])
        m.close()

        # Results
        r = h5py.File(result_path, 'r')
        self.result_path = result_path

        temp = ""
        if(self.single_state_path in r):
            for key in r[self.single_state_path].keys():
                temp = key
                while not (self.single_state_path + "/" + temp + "/entityresults/NODE/FILLING_FACTOR/ZONE1_set1/erfblock/res" in r):
                    temp = self.decrement(temp)
            temp = r["post/singlestate/" + temp + "/entityresults/NODE/FILLING_FACTOR/ZONE1_set1/erfblock/res"][()]
        if(len(temp) > 0):
            self.avg_level = np.sum(temp) / len(temp)

        if(re.search("([0-9]{4}\-[0-9]{2}\-[0-9]{2}_[0-9]{2}\-[0-9]{2}\-[0-9]{2})", result_path) != None):
            self.age = datetime.strptime(re.search("([0-9]{4}\-[0-9]{2}\-[0-9]{2}_[0-9]{2}\-[0-9]{2}\-[0-9]{2})", result_path).group(1), '%Y-%m-%d_%H-%M-%S')
        if(self.number_of_sensors_path in r):
            self.number_of_sensors = np.shape(r[self.number_of_sensors_path][()])[1]
        r.close()

    def decrement(self, s):
        lastNum = re.compile(r'(?:[^\d]*(\d+)[^\d]*)+')
        m = lastNum.search(s)
        if m:
            next = str(int(m.group(1))-1)
            start, end = m.span(1)
            s = s[:max(end-len(next), start)] + next + s[end:]
        return s

    def show_object_content(self):
        x = PrettyTable()
        x.field_names = ["Metaparameters", "Metadata"]
        x.add_row(["Path_Metadata", str(self.meta_path)])
        if(hasattr(self, 'output_frequency_type')):
            x.add_row(["Output_Frequency_Type", str(self.output_frequency_type)])

        if(hasattr(self, 'general_sigma')):
            x.add_row(["General_Sigma", str(self.general_sigma)])
        if(hasattr(self, 'number_of_circles')):
            x.add_row(["Number_Circles", str(self.number_of_circles)])
        if(hasattr(self, 'number_of_rectangles')):
            x.add_row(["Number_Rectangles", str(self.number_of_rectangles)])
        if(hasattr(self, 'number_of_runners')):
            x.add_row(["Number_Runners", str(self.number_of_runners)])
        if(hasattr(self, 'number_of_shapes')):
            x.add_row(["Number_Shapes", str(self.number_of_shapes)])
        if(hasattr(self, 'fibre_content_circles')):
            x.add_row(["Fibre_Content_Circles", str(self.fibre_content_circles)])
        if(hasattr(self, 'fibre_content_rectangles')):
            x.add_row(["Fibre_Content_Rectangles", str(self.fibre_content_rectangles)])
        if(hasattr(self, 'fibre_content_runners')):
            x.add_row(["Fibre_Content_Runners", str(self.fibre_content_runners)])

        if(hasattr(self, 'fvc_circle')):
            x.add_row(["FVC_Circle", str(self.fvc_circle)])
        if(hasattr(self, 'radius_circle')):
            x.add_row(["Radius_Circle", str(self.radius_circle)])
        if(hasattr(self, 'fvc_rectangle')):
            x.add_row(["FVC_Rectangle", str(self.fvc_rectangle)])
        if(hasattr(self, 'height_rectangle')):
            x.add_row(["Height_Rectangle", str(self.height_rectangle)])
        if(hasattr(self, 'width_rectangle')):
            x.add_row(["Width_Rectangle", str(self.width_rectangle)])
        if(hasattr(self, 'posx_rectangle')):
            x.add_row(["PosX_Rectangle", str(self.posx_rectangle)])

        # Result
        y = PrettyTable()
        y.field_names = ["Resultparameters", "Resultdata"]
        if(hasattr(self, 'result_path')):
            y.add_row(["     Path_Resultdata    ", str(self.result_path) + "  "]) 
        if(hasattr(self, 'avg_level')):
            y.add_row(["Average_Level", str(self.avg_level)])
        if(hasattr(self, 'age')):
            y.add_row(["Age of file", str(self.age)])
        if(hasattr(self, 'number_of_sensors')):
            y.add_row(["Number_Sensors", str(self.number_of_sensors)])
        print(x)
        print(y)

# test = HDF5Object("Data/2019-07-23_15-38-08_0")
# test.showObjectContent()
