import glob
import os
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import regex as re
from prettytable import PrettyTable


class HDF5Object:
    def __init__(self, metaPath, resultPath):
        self.outputFrequencyTypePath = "output_frequency_type"
        #Pertubations_Factors
        self.generalSigmaPath = "perturbation_factors/General_Sigma"
        #-Shapes
        self.numberOfCirclesPath = "perturbation_factors/Shapes/Circles/Num"
        self.numberOfRectanglesPath = "perturbation_factors/Shapes/Rectangles/Num"
        self.numberOfRunnersPath = "perturbation_factors/Shapes/Runners/Num"
        self.fibreContentCirclesPath = "perturbation_factors/Shapes/Circles/Fiber_Content"
        self.fibreContentRectanglesPath = "perturbation_factors/Shapes/Rectangles/Fiber_Content"
        self.fibreContentRunnersPath = "perturbation_factors/Shapes/Runners/Fiber_Content"
        #Shapes
        #-Circlepaths
        self.fvcCirclePath = "shapes/Circle/fvc"
        self.radiusCirclePath = "shapes/Circle/radius"
        self.posXCirclePath = "shapes/Circle/posX"
        self.posYCirclePath = "shapes/Circle/posY"
        #-Rectanglepaths
        self.fvcRectanglePath = "shapes/Rectangle/fvc"
        self.heightRectanglePath = "shapes/Rectangle/height"
        self.widthRectanglePath = "shapes/Rectangle/width"
        self.posXRectanglePath = "shapes/Rectangle/posX"
        self.posYRectanglePath = "shapes/Rectangle/posY"
        #-Runnerspaths
        self.fvcRunnerPath = "shapes/Runner/fvc"
        self.heightRunnerPath = "shapes/Runner/height"
        self.widthRunnerPath = "shapes/Runner/width"
        self.posXRunnerPath = "shapes/Runner/posX"
        self.posYRunnerPath = "shapes/Runner/posY"
        self.posLowerLeftXRunnerPath = "shapes/Runner/posLowerLeftX"
        self.posLowerLeftYRunnerPath = "shapes/Runner/posLowerLeftY"

        self.singleStatePath = "post/singlestate"
        self.numberOfSensorsPath = "post/multistate/TIMESERIES1/multientityresults/SENSOR/PRESSURE/ZONE1_set1/erfblock/res"

        self.numberOfCircles = 0
        self.numberOfRectangles = 0
        self.numberOfRunners = 0
        self.numberOfSensors = 0

        #Metadata
        m = h5py.File(metaPath, 'r')
        self.metaPath = metaPath
        if(self.outputFrequencyTypePath in m):
            self.outputFrequencyType = m[self.outputFrequencyTypePath][()]

        if(self.generalSigmaPath in m):
            self.generalSigma = m[self.generalSigmaPath][()]
        if(self.numberOfCirclesPath in m):
            self.numberOfCircles = m[self.numberOfCirclesPath][()]
        if(self.numberOfRectanglesPath in m):
            self.numberOfRectangles = m[self.numberOfRectanglesPath][()]
        if(self.numberOfRunnersPath in m):
            self.numberOfRunners = m[self.numberOfRunnersPath][()]
        self.numberOfShapes = self.numberOfCircles + self.numberOfRectangles + self.numberOfRunners
        if(self.fibreContentCirclesPath in m):
            self.fibreContentCircles = m[self.fibreContentCirclesPath][()]
        if(self.fibreContentRectanglesPath in m):
            self.fibreContentRectangles = m[self.fibreContentRectanglesPath][()]

        if(self.fibreContentRunnersPath in m):
            self.fibreContentRunners = m[self.fibreContentRunnersPath][()]

        #Shapes
        #-Circle
        if(self.fvcCirclePath in m):
            self.fvcCircle = np.array(m[self.fvcCirclePath][()])
        if(self.radiusCirclePath in m):
            self.radiusCircle = np.array(m[self.radiusCirclePath][()])
        if(self.posXCirclePath in m):
            self.posXCircle = np.array(m[self.posXCirclePath][()])
        if(self.posYCirclePath in m):
            self.posyCircle = np.array(m[self.posYCirclePath][()])
        #-Rectangle
        if(self.fvcRectanglePath in m):
            self.fvcRectangle = np.array(m[self.fvcRectanglePath][()])
        if(self.heightRectanglePath in m):
            self.heightRectangle = np.array(m[self.heightRectanglePath][()])
        if(self.widthRectanglePath in m):
            self.widthRectangle = np.array(m[self.widthRectanglePath][()])
        if(self.posXRectanglePath in m):
            self.posXRectangle = np.array(m[self.posXRectanglePath][()])
        if(self.posYRectanglePath in m):
            self.posYRectangle = np.array(m[self.posYRectanglePath][()])
        #-Runner
        if(self.fvcRunnerPath in m):
            self.fvcRunner = np.array(m[self.fvcRunnerPath][()])
        if(self.heightRunnerPath in m):
            self.heightRunner = np.array(m[self.heightRunnerPath][()])
        if(self.widthRunnerPath in m):
            self.widthRunner = np.array(m[self.widthRunnerPath][()])
        if(self.posXRunnerPath in m):
            self.posXRunner = np.array(m[self.posXRunnerPath][()])
        if(self.posYRunnerPath in m):
            self.posYRunner = np.array(m[self.posYRunnerPath][()])
        if(self.posLowerLeftXRunnerPath in m):
            self.posLowerLeftXRunner = np.array(m[self.posLowerLeftXRunnerPath][()])
        if(self.posLowerLeftYRunnerPath in m):
            self.posLowerLeftYRunner = np.array(m[self.posLowerLeftYRunnerPath][()])
        m.close()

        #Results
        r = h5py.File(resultPath, 'r')
        self.resultPath = resultPath

        temp = ""
        if(self.singleStatePath in r):
            for key in r[self.singleStatePath].keys():
                temp = key
                while not (self.singleStatePath + "/" + temp + "/entityresults/NODE/FILLING_FACTOR/ZONE1_set1/erfblock/res" in r):
                    temp = self.decrement(temp)
            temp = r["post/singlestate/" + temp + "/entityresults/NODE/FILLING_FACTOR/ZONE1_set1/erfblock/res"][()]
        self.avgLevel = np.sum(temp) / len(temp)

        self.age = datetime.strptime(re.search("([0-9]{4}\-[0-9]{2}\-[0-9]{2}_[0-9]{2}\-[0-9]{2}\-[0-9]{2})", resultPath).group(1), '%Y-%m-%d_%H-%M-%S')
        if(self.numberOfSensorsPath in r):
            self.numberOfSensors = np.shape(r[self.numberOfSensorsPath][()])[1]
        r.close()

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
        if(hasattr(self, 'outputFrequencyType')):
            x.add_row(["Output_Frequency_Type", str(self.outputFrequencyType)])

        if(hasattr(self, 'generalSigma')):
            x.add_row(["General_Sigma", str(self.generalSigma)])
        if(hasattr(self, 'numberOfCircles')):
            x.add_row(["Number_Circles", str(self.numberOfCircles)])
        if(hasattr(self, 'numberOfRectangles')):
            x.add_row(["Number_Rectangles", str(self.numberOfRectangles)])
        if(hasattr(self, 'numberOfRunners')):
            x.add_row(["Number_Runners", str(self.numberOfRunners)])
        if(hasattr(self, 'numberOfShapes')):
            x.add_row(["Number_Shapes", str(self.numberOfShapes)])
        if(hasattr(self, 'fibreContentCircles')):
            x.add_row(["Fibre_Content_Circles", str(self.fibreContentCircles)])
        if(hasattr(self, 'fibreContentRectangles')):
            x.add_row(["Fibre_Content_Rectangles", str(self.fibreContentRectangles)])
        if(hasattr(self, 'fibreContentRunners')):
            x.add_row(["Fibre_Content_Runners", str(self.fibreContentRunners)])

        if(hasattr(self, 'fvcCircle')):
            x.add_row(["FVC_Circle", str(self.fvcCircle)])
        if(hasattr(self, 'radiusCircle')):
            x.add_row(["Radius_Circle", str(self.radiusCircle)])
        if(hasattr(self, 'fvcRectangle')):
            x.add_row(["FVC_Rectangle", str(self.fvcRectangle)])
        if(hasattr(self, 'heightRectangle')):
            x.add_row(["Height_Rectangle", str(self.heightRectangle)])
        if(hasattr(self, 'widthRectangle')):
            x.add_row(["Width_Rectangle", str(self.widthRectangle)])
        if(hasattr(self, 'posXRectangle')):
            x.add_row(["PosX_Rectangle", str(self.posXRectangle)])
        
        #Result
        y = PrettyTable()
        y.field_names = ["Resultparameters", "Resultdata"]
        y.add_row(["     Path_Resultdata    ", str(self.resultPath) + "  "])        
        y.add_row(["Average_Level", str(self.avgLevel)])
        y.add_row(["Age of file", str(self.age)])
        y.add_row(["Number_Sensors", str(self.numberOfSensors)])
        print(x)
        print(y)
    
#test = HDF5Object("Data/2019-07-23_15-38-08_0")
#test.showObjectContent()
