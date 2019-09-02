import os
import pickle
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import regex as re
from prettytable import PrettyTable

from HDF5Object import HDF5Object


class HDF5DB:
    def __init__(self):
        self.hdf5_object_list = []

    def add_object(self, object):
        if(object != None):
            self.hdf5_object_list.append(object)

    def add_objects_from_path(self, path):
        dirpath = os.getcwd() / Path(path)
        if(dirpath.is_dir()):
            # List all hdft-files
            hdf5File = dirpath.rglob("**/*.hdf5")
            for i in hdf5File:
                # Check that only *.hdf5 and *.erfh5 files will be opened
                if(h5py.is_hdf5(i.as_posix())):
                    erfh5File = Path(str(i).replace("meta_data.hdf5", "RESULT.erfh5"))
                    if(erfh5File.exists()):
                        self.hdf5_object_list.append(HDF5Object(i.as_posix(), erfh5File.as_posix()))
                    else:
                        print(erfh5File.as_posix() + " does not exist. The folder was skipped.")
                else:
                    print(i.as_posix() + " does not exist. The folder was skipped.")
            print(str(len(self.hdf5_object_list)) + " Objects have been added.")
        else:
            print("The path " + path + " does not exist! No objects were added!")

    def select(self, variable, value, comparisonOperator="="):
        self.selected = []
        for obj in self.hdf5_object_list:
            if (comparisonOperator == "="):
                if (variable == "meta_path" and obj.meta_path == value):
                    self.selected.append(obj)
                elif (variable == "output_frequency_type" and obj.output_frequency_type == value):
                    self.selected.append(obj)
                elif (variable == "general_sigma" and obj.general_Sigma == value):
                    self.selected.append(obj)
                elif (variable == "number_of_circles" and obj.number_of_circles == value):
                    self.selected.append(obj)
                elif (variable == "number_of_rectangles" and obj.number_of_rectangles == value):
                    self.selected.append(obj)
                elif (variable == "number_of_runners" and obj.number_of_runners == value):
                    self.selected.append(obj)
                elif (variable == "number_of_shapes" and obj.number_of_shapes == value):
                    self.selected.append(obj)
                elif (variable == "fibre_content_circles" and np.amin(obj.fibre_content_circles) <= value and np.amax(obj.fibre_content_circles) >= value):
                    self.selected.append(obj)
                elif (variable == "fibre_content_rectangles" and np.amin(obj.fibre_content_rectangles) <= value and np.amax(obj.fibre_content_rectangles) >= value):
                    self.selected.append(obj)
                elif (variable == "fibre_content_runners" and np.amin(obj.fibre_content_runners) <= value and np.amax(obj.fibre_content_runners) >= value):
                    self.selected.append(obj)
                # Circle
                elif (variable == "fvc_circle" and value in obj.fvc_circle):
                    self.selected.append(obj)
                elif (variable == "radius_circle" and value in obj.radius_circle):
                    self.selected.append(obj)
                elif (variable == "pox_circle" and value in obj.posx_circle):
                    self.selected.append(obj)
                elif (variable == "posy_circle" and value in obj.posy_circle):
                    self.selected.append(obj)
                # Rectangle
                elif (variable == "fvc_rectangle" and value in obj.fvc_rectangle):
                    self.selected.append(obj)
                elif (variable == "height_rectangle" and value in obj.height_rectangle):
                    self.selected.append(obj)
                elif (variable == "width_rectangle" and value in obj.width_rectangle):
                    self.selected.append(obj)
                elif (variable == "posx_rectangle" and value in obj.posx_rectangle):
                    self.selected.append(obj)
                elif (variable == "posy_rectangle" and value in obj.posy_rectangle):
                    self.selected.append(obj)
                # Runner
                elif (variable == "fvc_runner" and value in obj.fvc_runner):
                    self.selected.append(obj)
                elif (variable == "height_runner" and value in obj.height_runner):
                    self.selected.append(obj)
                elif (variable == "width_runner" and value in obj.width_runner):
                    self.selected.append(obj)
                elif (variable == "posx_runner" and value in obj.posx_runner):
                    self.selected.append(obj)
                elif (variable == "posy_runner" and value in obj.posy_runner):
                    self.selected.append(obj)
                elif (variable == "pos_lower_left__runner" and value in obj.pos_lower_leftx_runner):
                    self.selected.append(obj)
                elif (variable == "pos_lower_lefty_runner" and value in obj.pos_lower_lefty_runner):
                    self.selected.append(obj)
                # Result
                elif (variable == "result_path" and obj.result_path == value):
                    self.selected.append(obj)
                elif (variable == "avg_level" and obj.avg_level == value):
                    self.selected.append(obj)
                elif (variable == "age" and obj.age == datetime.strptime(re.search("([0-9]{4}\-[0-9]{2}\-[0-9]{2}_[0-9]{2}\-[0-9]{2}\-[0-9]{2})", value).group(1), '%Y-%m-%d_%H-%M-%S')):
                    self.selected.append(obj)
                elif (variable == "number_of_sensors" and obj.number_of_sensors == value):
                    self.selected.append(obj)

            elif (comparisonOperator == ">"):
                if (variable == "meta_path"):
                    print("The operator " + comparisonOperator + " is not available for metaPath.")
                    return
                elif (variable == "output_frequency_type" and obj.output_frequency_type > value):
                    self.selected.append(obj)
                elif (variable == "general_sigma" and obj.general_sigma > value):
                    self.selected.append(obj)
                elif (variable == "number_of_circles" and obj.number_of_circles > value):
                    self.selected.append(obj)
                elif (variable == "number_of_rectangles" and obj.number_of_rectangles > value):
                    self.selected.append(obj)
                elif (variable == "number_of_runners" and obj.number_of_runners > value):
                    self.selected.append(obj)
                elif (variable == "number_of_shapes" and obj.number_of_shapes > value):
                    self.selected.append(obj)
                elif (variable == "fibre_content_circles" and np.amin(obj.fibre_content_circles) > value):
                    self.selected.append(obj)
                elif (variable == "fibre_content_rectangles" and np.amin(obj.fibre_content_rectangles) > value):
                    self.selected.append(obj)
                elif (variable == "fibre_content_runners" and np.amin(obj.fibre_content_runners) > value):
                    self.selected.append(obj)
                # Circle
                elif (variable == "fvc_circle" and np.amin(obj.fvc_circle) > value):
                    self.selected.append(obj)
                elif (variable == "radius_circle" and np.amin(obj.radius_circle) > value):
                    self.selected.append(obj)
                elif (variable == "posx_circle" and np.amin(obj.posx_circle) > value):
                    self.selected.append(obj)
                elif (variable == "posy_circle" and np.amin(obj.posy_circle) > value):
                    self.selected.append(obj)
                # Rectangle
                elif (variable == "fvc_rectangle" and np.amin(obj.fvc_rectangle) > value):
                    self.selected.append(obj)
                elif (variable == "height_rectangle" and np.amin(obj.height_rectangle) > value):
                    self.selected.append(obj)
                elif (variable == "width_rectangle" and np.amin(obj.width_rectangle) > value):
                    self.selected.append(obj)
                elif (variable == "posx_rectangle" and np.amin(obj.posx_rectangle) > value):
                    self.selected.append(obj)
                elif (variable == "posy_rectangle" and np.amin(obj.posy_rectangle) > value):
                    self.selected.append(obj)
                # Runner
                elif (variable == "fvc_runner" and np.amin(obj.fvc_runner) > value):
                    self.selected.append(obj)
                elif (variable == "height_runner" and np.amin(obj.height_runner) > value):
                    self.selected.append(obj)
                elif (variable == "width_runner" and np.amin(obj.width_runner) > value):
                    self.selected.append(obj)
                elif (variable == "posx_runner" and np.amin(obj.posx_runner) > value):
                    self.selected.append(obj)
                elif (variable == "posy_runner" and np.amin(obj.posy_runner) > value):
                    self.selected.append(obj)
                elif (variable == "pos_lower_leftx_runner" and np.amin(obj.pos_lower_leftx_runner) > value):
                    self.selected.append(obj)
                elif (variable == "pos_lower_lefty_runner" and np.amin(obj.pos_lower_lefty_runner) > value):
                    self.selected.append(obj)
                # Result
                elif (variable == "result_path"):
                    print("The operator " + comparisonOperator + " is not available for resultPath.")
                    return
                elif (variable == "avg_level" and obj.avg_level > value):
                    self.selected.append(obj)
                elif (variable == "age" and obj.age > datetime.strptime(re.search("([0-9]{4}\-[0-9]{2}\-[0-9]{2}_[0-9]{2}\-[0-9]{2}\-[0-9]{2})", value).group(1), '%Y-%m-%d_%H-%M-%S')):
                    self.selected.append(obj)
                elif (variable == "number_of_sensors" and obj.number_of_sensors > value):
                    self.selected.append(obj)

            elif (comparisonOperator == "<"):
                if (variable == "meta_path"):
                    print("The operator " + comparisonOperator + " is not available for metaPath.")
                    return
                elif (variable == "output_frequency_type" and obj.output_frequency_type < value):
                    self.selected.append(obj)
                elif (variable == "general_sigma" and obj.general_sigma < value):
                    self.selected.append(obj)
                elif (variable == "number_of_circles" and obj.number_of_circles < value):
                    self.selected.append(obj)
                elif (variable == "number_of_rectangles" and obj.number_of_rectangles < value):
                    self.selected.append(obj)
                elif (variable == "number_of_runners" and obj.number_of_runners < value):
                    self.selected.append(obj)
                elif (variable == "number_of_shapes" and obj.number_of_shapes < value):
                    self.selected.append(obj)
                elif (variable == "fibre_content_circles" and np.amax(obj.fibre_content_circles) < value):
                    self.selected.append(obj)
                elif (variable == "fibre_content_rectangles" and np.amax(obj.fibre_content_rectangles) < value):
                    self.selected.append(obj)
                elif (variable == "fibre_content_runners" and np.amax(obj.fibre_content_runners) < value):
                    self.selected.append(obj)
                # Circle
                elif (variable == "fvc_circle" and np.amax(obj.fvc_circle) < value):
                    self.selected.append(obj)
                elif (variable == "radius_circle" and np.amax(obj.radius_circle) < value):
                    self.selected.append(obj)
                elif (variable == "posx_circle" and np.amax(obj.posx_circle) < value):
                    self.selected.append(obj)
                elif (variable == "posy_circle" and np.amax(obj.posy_circle) < value):
                    self.selected.append(obj)
                # Rectangle
                elif (variable == "fvc_rectangle" and np.amax(obj.fvc_rectangle) < value):
                    self.selected.append(obj)
                elif (variable == "height_rectangle" and np.amax(obj.height_rectangle) < value):
                    self.selected.append(obj)
                elif (variable == "width_rectangle" and np.amax(obj.width_rectangle) < value):
                    self.selected.append(obj)
                elif (variable == "posx_rectangle" and np.amax(obj.posx_rectangle) < value):
                    self.selected.append(obj)
                elif (variable == "posy_rectangle" and np.amax(obj.posy_rectangle) < value):
                    self.selected.append(obj)
                # Runner
                elif (variable == "fvc_runner" and np.amax(obj.fvc_runner) < value):
                    self.selected.append(obj)
                elif (variable == "height_runner" and np.amax(obj.height_runner) < value):
                    self.selected.append(obj)
                elif (variable == "width_runner" and np.amax(obj.width_runner) < value):
                    self.selected.append(obj)
                elif (variable == "posx_runner" and np.amax(obj.posx_runner) < value):
                    self.selected.append(obj)
                elif (variable == "posy_runner" and np.amax(obj.posy_runner) < value):
                    self.selected.append(obj)
                elif (variable == "pos_lower_leftx_runner" and np.amax(obj.pos_lower_leftx_runner) < value):
                    self.selected.append(obj)
                elif (variable == "pos_lower_lefty_runner" and np.amax(obj.pos_lower_lefty_runner) < value):
                    self.selected.append(obj)
                # Result
                elif (variable == "result_path"):
                    print("The operator " + comparisonOperator + " is not available for resultPath.")
                    return
                elif (variable == "avg_level" and obj.avg_level < value):
                    self.selected.append(obj)
                elif (variable == "age" and obj.age < datetime.strptime(re.search("([0-9]{4}\-[0-9]{2}\-[0-9]{2}_[0-9]{2}\-[0-9]{2}\-[0-9]{2})", value).group(1), '%Y-%m-%d_%H-%M-%S')):
                    self.selected.append(obj)
                elif (variable == "number_of_sensors" and obj.number_of_sensors < value):
                    self.selected.append(obj)

        if(len(self.selected) == 0):
            print("No matches were found for " + str(variable) + " and parameter " + str(value) + ". No filters were applied!")
        else:
            self.HDF5Object = self.selected
            print("\nThe filter " + str(variable) + " " + str(comparisonOperator) + " " + str(value) + " was applied. " + str(len(self.selected)) + " objects were found.")
            self.hdf5_object_list =  self.selected

    def show_selection_options(self):
        self.options = PrettyTable()
        self.options.field_names = ["Possible options are"]
        self.options.add_row(["meta_path"])
        self.options.add_row(["output_frequency_type"])
        self.options.add_row(["general_sigma"])
        self.options.add_row(["number_of_circles"])
        self.options.add_row(["number_of_rectangles"])
        self.options.add_row(["number_of_runners"])
        self.options.add_row(["number_of_shapes"])
        self.options.add_row(["fibre_content_circles"])
        self.options.add_row(["fibre_content_rectangles"])
        self.options.add_row(["fibre_content_runners"])
        self.options.add_row(["fvc_circle"])
        self.options.add_row(["radius_circle"])
        self.options.add_row(["posx_circle"])
        self.options.add_row(["posY_circle"])
        self.options.add_row(["fvc_rectangle"])
        self.options.add_row(["height_rectangle"])
        self.options.add_row(["width_rectangle"])
        self.options.add_row(["posX_rectangle"])
        self.options.add_row(["posY_rectangle"])
        self.options.add_row(["fvc_runner"])
        self.options.add_row(["height_runner"])
        self.options.add_row(["width_runner"])
        self.options.add_row(["posX_runner"])
        self.options.add_row(["posY_runner"])
        self.options.add_row(["pos_lower_leftx_runner"])
        self.options.add_row(["pos_lower_lefty_runner"])
        self.options.add_row(["result_path"])
        self.options.add_row(["avg_level"])
        self.options.add_row(["age"])
        self.options.add_row(["number_of_sensors"])
        print(self.options)

    def show_objects(self):
        if not(len(self.hdf5_object_list) == 0):
            for val in self.hdf5_object_list:
                val.show_object_content()
                print("\n")
        else:
            print("No objects were found.")

    def save(self, path, filename="HDF5DB"):
        dirpath = Path(path)
        if(dirpath.is_dir()):
            if not(len(self.hdf5_object_list) == 0):
                file = Path(filename + r'.h5db')
                h5db_path = dirpath / file
                if(h5db_path.is_file()):
                    print("A file with the given name already exists. " + filename + " will be overwritten. Do you want to continue?\nPlease type in yes to continue.")
                    userinput = input("")
                    if not(userinput == "yes"):
                        print("Nothing has been saved.")
                        return
                    else:
                        print(filename + " will be overwritten.")
                outfile = open(dirpath / file, "wb")
                pickle.dump(self.hdf5_object_list, outfile)
                outfile.close()
                print("HDF5DB saved")
            else:
                print("No objects were found. Nothing was saved!")
        else:
            print(path + " does not exist! Nothing was saved!")

    def load(self, path, filename="HDF5DB"):
        dir_path = Path(path)
        h5db_path = dir_path / Path(filename + str(".h5db"))
        if(h5db_path.is_file()):
            infile = open(dir_path / Path(filename + ".h5db"), "rb")
            self.hdf5_object_list = pickle.load(infile)
            infile.close()
            print("HDF5DB loaded")
        else:
            print("There is no h5db-file at the given path with the given name!")
