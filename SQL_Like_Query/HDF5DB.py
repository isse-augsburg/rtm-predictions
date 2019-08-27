import h5py
import pickle
import os
import regex as re
from datetime import datetime
from prettytable import PrettyTable
from glob import glob
import numpy as np
from HDF5Object import HDF5Object

#ToDo:
#       Testklasse für HDF5DB sowie HDF5Object
#       pathlib
class HDF5DB:
    def __init__(self):
        self.hdf5ObjectList = []
       
    def addObject(self, object):
        if(object != None):
            self.hdf5ObjectList.append(object)

    def addObjectsFromPath(self, path):
        if(os.path.exists(path)):
            for folder in os.listdir(path):            
                for hdf5File in glob(path + "/" + folder + "/*.hdf5"):
                    for erfh5File in glob(path + "/" + folder + "/*.erfh5"):
                        # Check that only *.hdf5 and *.erfh5 files will be opened
                        if (h5py.is_hdf5(hdf5File) and h5py.is_hdf5(erfh5File)):
                            #print(path + "/" + folder)
                            self.hdf5ObjectList.append(HDF5Object(path + "/" + folder))
                            
                    if not (h5py.is_hdf5(hdf5File)):
                        print(hdf5File + " is corrupt!\nThe file has been skipped.")                                                                                                            #"There is one or more corrupt files in " + path + "/" + folder + "!\n" + "The folder has been skipped.")
                    if not (h5py.is_hdf5(erfh5File)):
                        print(erfh5File + " is corrupt!\nThe file has been skipped.")
            print(str(len(self.hdf5ObjectList)) + " Objects have been added.")
        else:
            print("The path " + path + " does not exist! No objects were added!")

    def select(self, variable, value, comparisonOperator="="):
        self.selected = []
        for obj in self.hdf5ObjectList:
            if (comparisonOperator == "="):
                if (variable == "metaPath" and obj.metaPath == value):
                    self.selected.append(obj)
                elif (variable == "outputFrequencyType" and obj.outputFrequencyType == value):
                    self.selected.append(obj)
                elif (variable == "generalSigma" and obj.generalSigma == value):
                    self.selected.append(obj)
                elif (variable == "numberOfCircles" and obj.numberOfCircles == value):
                    self.selected.append(obj)
                elif (variable == "numberOfRectangles" and obj.numberOfRectangles == value):
                    self.selected.append(obj)
                elif (variable == "numberOfRunners" and obj.numberOfRunners == value):
                    self.selected.append(obj)
                elif (variable == "numberOfShapes" and obj.numberOfShapes == value):
                    self.selected.append(obj)
                elif (variable == "fibreContentCircles" and np.amin(obj.fibreContentCircles) <= value and np.amax(obj.fibreContentCircles) >= value):
                    self.selected.append(obj)
                elif (variable == "fibreContentRectangles" and np.amin(obj.fibreContentRectangles) <= value and np.amax(obj.fibreContentRectangles) >= value):
                    self.selected.append(obj)
                elif (variable == "fibreContentRunners" and np.amin(obj.fibreContentRunners) <= value and np.amax(obj.fibreContentRunners) >= value):
                    self.selected.append(obj)
                    #print("Min: " + str(np.amin(obj.fibreContentRunners)) + ", Max: " + str(np.amax(obj.fibreContentRunners)))
                elif (variable == "fvcCircle" and value in obj.fvcCircle):
                    self.selected.append(obj)
                elif (variable == "radiusCircle" and value in obj.radiusCircle):
                    self.selected.append(obj)
                elif (variable == "vcRectangle" and value in obj.vcRectangle):
                    self.selected.append(obj)
                elif (variable == "heightRectangle" and value in obj.heightRectangle):
                    self.selected.append(obj)
                elif (variable == "widthRectangle" and value in obj.widthRectangle):#np.isin(value, obj.widthRectangle)):
                    self.selected.append(obj)
                elif (variable == "resultPath" and obj.resultPath == value):
                    self.selected.append(obj)
                elif (variable == "avgLevel" and obj.avgLevel == value):
                    self.selected.append(obj)
                elif (variable == "age" and obj.age == datetime.strptime(re.search("([0-9]{4}\-[0-9]{2}\-[0-9]{2}_[0-9]{2}\-[0-9]{2}\-[0-9]{2})", value).group(1), '%Y-%m-%d_%H-%M-%S')):
                    self.selected.append(obj)
                elif (variable == "numberOfSensors" and obj.numberOfSensors == value):
                    self.selected.append(obj)

            elif (comparisonOperator == ">"):
                if (variable == "metaPath"):
                    print("The operator " + comparisonOperator + " is not available for metaPath.")
                    break
                elif (variable == "outputFrequencyType" and obj.outputFrequencyType > value):
                    self.selected.append(obj)
                elif (variable == "generalSigma" and obj.generalSigma > value):
                    self.selected.append(obj)
                elif (variable == "numberOfCircles" and obj.numberOfCircles > value):
                    self.selected.append(obj)
                elif (variable == "numberOfRectangles" and obj.numberOfRectangles > value):
                    self.selected.append(obj)
                elif (variable == "numberOfRunners" and obj.numberOfRunners > value):
                    self.selected.append(obj)
                elif (variable == "numberOfShapes" and obj.numberOfShapes > value):
                    self.selected.append(obj)
                elif (variable == "fibreContentCircles" and np.amin(obj.fibreContentCircles) > value):
                    self.selected.append(obj)
                elif (variable == "fibreContentRectangles" and np.amin(obj.fibreContentRectangles) > value):
                    self.selected.append(obj)
                elif (variable == "fibreContentRunners" and np.amin(obj.fibreContentRunners) > value):
                    self.selected.append(obj)
                elif (variable == "fvcCircle" and np.amin(obj.fvcCircle) > value):
                    self.selected.append(obj)
                elif (variable == "radiusCircle" and np.amin(obj.radiusCircle) > value):
                    self.selected.append(obj)
                elif (variable == "vcRectangle" and np.amin(obj.vcRectangle) > value):
                    self.selected.append(obj)
                elif (variable == "heightRectangle" and np.amin(obj.heightRectangle) > value):
                    self.selected.append(obj)
                elif (variable == "widthRectangle" and np.amin(obj.widthRectangle) > value):
                    self.selected.append(obj)
                elif (variable == "resultPath"):
                    print("The operator " + comparisonOperator + " is not available for resultPath.")
                    break
                elif (variable == "avgLevel" and obj.avgLevel > value):
                    self.selected.append(obj)
                elif (variable == "age" and obj.age > datetime.strptime(re.search("([0-9]{4}\-[0-9]{2}\-[0-9]{2}_[0-9]{2}\-[0-9]{2}\-[0-9]{2})", value).group(1), '%Y-%m-%d_%H-%M-%S')):
                    self.selected.append(obj)
                elif (variable == "numberOfSensors" and obj.numberOfSensors > value):
                    self.selected.append(obj)

            elif (comparisonOperator == "<"):
                if (variable == "metaPath"):
                    print("The operator " + comparisonOperator + " is not available for metaPath.")
                    break
                elif (variable == "outputFrequencyType" and obj.outputFrequencyType < value):
                    self.selected.append(obj)
                elif (variable == "generalSigma" and obj.generalSigma < value):
                    self.selected.append(obj)
                elif (variable == "numberOfCircles" and obj.numberOfCircles < value):
                    self.selected.append(obj)
                elif (variable == "numberOfRectangles" and obj.numberOfRectangles < value):
                    self.selected.append(obj)
                elif (variable == "numberOfRunners" and obj.numberOfRunners < value):
                    self.selected.append(obj)
                elif (variable == "numberOfShapes" and obj.numberOfShapes < value):
                    self.selected.append(obj)
                elif (variable == "fibreContentCircles" and np.amax(obj.fibreContentCircles) < value):
                    self.selected.append(obj)
                elif (variable == "fibreContentRectangles" and np.amax(obj.fibreContentRectangles) < value):
                    self.selected.append(obj)
                elif (variable == "fibreContentRunners" and np.amax(obj.fibreContentRunners) < value):
                    self.selected.append(obj)
                elif (variable == "fvcCircle" and np.amax(obj.fvcCircle) < value):
                    self.selected.append(obj)
                elif (variable == "radiusCircle" and np.amax(obj.radiusCircle) < value):
                    self.selected.append(obj)
                elif (variable == "vcRectangle" and np.amax(obj.vcRectangle) < value):
                    self.selected.append(obj)
                elif (variable == "heightRectangle" and np.amax(obj.heightRectangle) < value):
                    self.selected.append(obj)
                elif (variable == "widthRectangle" and np.amax(obj.widthRectangle) < value):
                    self.selected.append(obj)
                elif (variable == "resultPath"):
                    print("The operator " + comparisonOperator + " is not available for resultPath.")
                    break
                elif (variable == "avgLevel" and obj.avgLevel < value):
                    self.selected.append(obj)
                elif (variable == "age" and obj.age < datetime.strptime(re.search("([0-9]{4}\-[0-9]{2}\-[0-9]{2}_[0-9]{2}\-[0-9]{2}\-[0-9]{2})", value).group(1), '%Y-%m-%d_%H-%M-%S')):
                    self.selected.append(obj)
                elif (variable == "numberOfSensors" and obj.numberOfSensors < value):
                    self.selected.append(obj)

        if (len(self.selected) == 0):
            print("No matches were found for " + str(variable) + " and parameter " + str(value) + ". No filters were applied!")
        else:
            self.HDF5Object = self.selected
            print("The filter " + str(variable) + " " + str(comparisonOperator) + " " + str(value) + " was applied. " + str(len(self.selected)) + " objects were found.")
            self.hdf5ObjectList =  self.selected

    def showSelectionOptions(self):
        self.options = PrettyTable()
        self.options.field_names = ["Possible options are"]
        self.options.add_row(["metaPath"])
        self.options.add_row(["outputFrequencyType"])
        self.options.add_row(["generalSigma"])
        self.options.add_row(["numberOfCircles"])
        self.options.add_row(["numberOfRectangles"])
        self.options.add_row(["numberOfRunners"])
        self.options.add_row(["numberOfShapes"])
        self.options.add_row(["fibreContentCircles"])
        self.options.add_row(["fibreContentRectangles"])
        self.options.add_row(["fibreContentRunners"])
        self.options.add_row(["fvcCircle"])
        self.options.add_row(["radiusCircle"])
        self.options.add_row(["fvcRectangle"])
        self.options.add_row(["heightRectangle"])
        self.options.add_row(["widthRectangle"])
        self.options.add_row(["resultPath"])
        self.options.add_row(["avgLevel"])
        self.options.add_row(["age"])
        self.options.add_row(["numberOfSensors"])
        print(self.options)

    def showObjects(self):
        if not(len(self.hdf5ObjectList) == 0):
            for val in self.hdf5ObjectList:
                val.showObjectContent()
                print("\n")
        else:
            print("No objects were found.")

    def save(self, path, filename="HDF5DB"):
        if(os.path.exists(path)):
            if not(len(self.hdf5ObjectList) == 0):
                if(os.path.isfile(path + "/" + filename + ".h5db")):
                    print(filename + " will be overwritten. Do you want to continue?\nPlease type in yes to continue.")
                    userinput = input("")
                    if not(userinput == "yes"):
                        print("Nothing has been saved.")
                        return
                    else:
                        print(filename + " will be overwritten.")
                if not(path == ""):
                    path = path + "/"                
                outfile = open(path + filename +  ".h5db", "wb" )
                pickle.dump(self.hdf5ObjectList, outfile)
                outfile.close()
                print("HDF5DB saved")
            else:
                print("No objects were found. Nothing was saved!")
        else:
            print(path + " does not exist! Nothing was saved!")

    def load(self, path, filename="HDF5DB"):
        if not(path == ""):
            path = path + "/"
        if(os.path.isfile(path + filename + ".h5db")):
            infile = open(path + "HDF5DB.h5db", "rb" )
            self.hdf5ObjectList = pickle.load(infile)
            infile.close()
            print("HDF5DB loaded")
        else:
            print("There is no h5db-file at the given path with the given name!")