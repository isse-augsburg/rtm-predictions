import h5py
import numpy as np
import regex as re
import glob
from datetime import datetime
from prettytable import PrettyTable
import os


class HDF5Object:
    def __init__(self, path):
        meta = "_meta_data.hdf5"
        result = "_RESULT.erfh5"

        #Metadata
        self.metaPath = ""
        for file in glob.glob(path+"/*.hdf5"):
            self.metaPath = file

        m = h5py.File(self.metaPath, 'r')
        self.outputFrequencyType = m["output_frequency_type"][()]

        self.generalSigma = m["perturbation_factors/General_Sigma"][()]
        self.numberOfCircles = m["perturbation_factors/Shapes/Circles/Num"][()]
        self.numberOfRectangles = m["perturbation_factors/Shapes/Rectangles/Num"][()]
        self.numberOfRunners = m["perturbation_factors/Shapes/Runners/Num"][()]
        self.numberOfShapes = self.numberOfCircles + self.numberOfRectangles + self.numberOfRunners
        self.fibreContentCircles = m["perturbation_factors/Shapes/Circles/Fiber_Content"][()]
        self.fibreContentRectangles = m["perturbation_factors/Shapes/Rectangles/Fiber_Content"][()]
        self.fibreContentRunners = m["perturbation_factors/Shapes/Runners/Fiber_Content"][()]

        self.fvcCircle = np.array(m["shapes/Circle/fvc"][()])
        self.radiusCircle = np.array(m["shapes/Circle/radius"][()])
        self.fvcRectangle = np.array(m["shapes/Rectangle/fvc"][()])
        self.heightRectangle = np.array(m["shapes/Rectangle/height"][()])
        self.widthRectangle = np.array(m["shapes/Rectangle/width"][()])
        m.close()

        #Results
        self.resultPath = ""
        for file in glob.glob(path+"/*.erfh5"):
            self.resultPath = file

        r = h5py.File(self.resultPath, 'r')

        temp = ""
        for key in r["post/singlestate"].keys():
            temp = key
        while not ("post/singlestate/" + temp + "/entityresults/NODE/FILLING_FACTOR/ZONE1_set1/erfblock/res" in r):
            temp = self.decrement(temp)
        temp = r["post/singlestate/" + temp + "/entityresults/NODE/FILLING_FACTOR/ZONE1_set1/erfblock/res"][()]
        self.avgLevel = np.sum(temp) / len(temp) 

        self.age = datetime.strptime(re.search("([0-9]{4}\-[0-9]{2}\-[0-9]{2}_[0-9]{2}\-[0-9]{2}\-[0-9]{2})", self.resultPath).group(1), '%Y-%m-%d_%H-%M-%S')
        self.numberOfSensors = np.shape(r["post/multistate/TIMESERIES1/multientityresults/SENSOR/PRESSURE/ZONE1_set1/erfblock/res"][()])[1]
        #self.sensorgrid
        #self.model
        r.close()
        #print(self.level)

    def decrement(self, s):
        lastNum = re.compile(r'(?:[^\d]*(\d+)[^\d]*)+')
        m = lastNum.search(s)
        if m:
            next = str(int(m.group(1))-1)
            start, end = m.span(1)
            s = s[:max(end-len(next), start)] + next + s[end:]
        return s

    def showObjectContent(self):
        x = PrettyTable()
        x.field_names = ["Metaparameters", "Metadata"]
        x.add_row(["Path_Metadata", str(self.metaPath)])
        x.add_row(["Output_Frequency_Type", str(self.outputFrequencyType)])

        x.add_row(["General_Sigma", str(self.generalSigma)])
        x.add_row(["Number_Circles", str(self.numberOfCircles)])
        x.add_row(["Number_Rectangles", str(self.numberOfRectangles)])
        x.add_row(["Number_Runners", str(self.numberOfRunners)])
        x.add_row(["Number_Shapes", str(self.numberOfShapes)])
        x.add_row(["Fibre_Content_Circles", str(self.fibreContentCircles)])
        x.add_row(["Fibre_Content_Rectangles", str(self.fibreContentRectangles)])
        x.add_row(["Fibre_Content_Runners", str(self.fibreContentRunners)])

        x.add_row(["FVC_Circle", str(self.fvcCircle)])
        x.add_row(["Radius_Circle", str(self.radiusCircle)])
        x.add_row(["FVC_Rectangle", str(self.fvcRectangle)])
        x.add_row(["Height_Rectangle", str(self.heightRectangle)])
        x.add_row(["Width_Rectangle", str(self.widthRectangle)])
        
        #Result
        y = PrettyTable()
        y.field_names = ["Resultparameters", "Resultdata"]
        y.add_row(["     Path_Resultdata    ", str(self.resultPath) + "  "])        
        y.add_row(["Average_Level", str(self.avgLevel)])
        y.add_row(["Age of file", str(self.age)])
        y.add_row(["Number_Sensors", str(self.numberOfSensors)])
        print(x)
        print(y)
        #print("" + self.sensorgrid)
        #print("" + self.model)
    
#test = HDF5Object("Data/2019-07-23_15-38-08_0")
#test.showObjectContent()