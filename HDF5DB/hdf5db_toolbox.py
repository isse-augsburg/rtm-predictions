import operator
import os
import pickle
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np
import regex as re
from prettytable import PrettyTable
from tqdm import tqdm

from HDF5DB.hdf5db_object import HDF5Object


class HDF5DBToolbox:
    def __init__(self):
        self.hdf5_object_list = []
        self.meta = []
        self.result = []

    def add_object(self, object):
        if object is not None:
            self.hdf5_object_list.append(object)

    def add_objects_from_path(self, path):
        self.get_meta_path_list()
        self.temp = len(self.hdf5_object_list)
        self.erfh5_path = []
        self.hdf5_path = []
        self.newObjects = []
        dirpath = os.getcwd() / Path(path)
        if dirpath.is_dir():
            # List all hdf5-files
            # hdf5File = dirpath.rglob("**/*.hdf5")
            print("Data is being retrieved...")
            # hdf5File_list = [el for el in tqdm(dirpath.rglob("**/*.hdf5"))]
            for i in tqdm([el for el in tqdm(dirpath.rglob("**/*.hdf5"))]):
                # Check that only *.hdf5 and *.erfh5 files will be opened
                if h5py.is_hdf5(i.as_posix()):
                    erfh5File = Path(str(i).replace("meta_data.hdf5", "RESULT.erfh5"))
                    if erfh5File.exists():
                        self.hdf5_path.append(i.as_posix())
                        self.erfh5_path.append(erfh5File.as_posix())
                        # self.newObject = HDF5Object(i.as_posix(), erfh5File.as_posix())
                        # if self.newObject.path_meta not in self.meta:
                        #     self.hdf5_object_list.append(self.newObject)
                else:
                    print(i.as_posix() + " does not exist. The folder was skipped.")
            print("H5-files are currently being scanned...")
            with Pool(20) as p:
                self.newObjects = p.starmap(
                    HDF5Object, zip(self.hdf5_path, self.erfh5_path)
                )
            print("Objects are currently being added...")
            for i in self.newObjects:
                if i.path_meta not in self.meta:
                    self.hdf5_object_list.append(i)
            print(
                str(abs(self.temp - len(self.hdf5_object_list)))
                + " Objects have been added."
            )
        else:
            print("The path " + path + " does not exist! No objects were added!")

    def select(self, variable, comparisonOperator, value):
        my_variable = []
        my_value = []
        my_comparison_operator = []
        selected = []
        if comparisonOperator not in ["=", ">", "<"]:
            print("The operator is unfortunately not available. Nothing was selected!")
            return -1
        for i in range(len(self.hdf5_object_list)):
            my_variable.append(variable)
            my_value.append(value)
            my_comparison_operator.append(comparisonOperator)

        for (a, b, c, d) in zip(
                self.hdf5_object_list, my_variable, my_comparison_operator, my_value
        ):
            selected.append(self.select_per_object(a, b, c, d))
        # with Pool(6) as p:
        #     selected = p.starmap(
        #         self.select, zip(self.hdf5_object_list, my_variable, my_value, my_comparison_operator)
        #     )
        selected = [a for a in selected if a is not None]

        if len(selected) == 0:
            print(
                "No matches were found for "
                + str(variable)
                + " "
                + str(comparisonOperator)
                + " "
                + str(value)
                + ". No filter was applied!"
            )
            return -1
        else:
            self.hdf5_object_list = selected
            if len(selected) > 1:
                print(
                    "The filter "
                    + str(variable)
                    + " "
                    + str(comparisonOperator)
                    + " "
                    + str(value)
                    + " was applied. "
                    + str(len(selected))
                    + " objects were found."
                )
            else:
                print(
                    "The filter "
                    + str(variable)
                    + " "
                    + str(comparisonOperator)
                    + " "
                    + str(value)
                    + " was applied. "
                    + str(len(selected))
                    + " object was found."
                )
            self.meta = [obj.path_meta for obj in self.hdf5_object_list]
            self.result = [obj.path_result for obj in self.hdf5_object_list]
            return 1

    def select_per_object(self, obj, variable, comparisonOperator, value):
        if hasattr(obj, str(variable)):
            if comparisonOperator == "=":
                operator1 = operator.eq
                operator2 = operator.contains
            elif comparisonOperator == ">":
                operator1 = operator.gt
                operator2 = np.amin
            elif comparisonOperator == "<":
                operator1 = operator.lt
                operator2 = np.amax
            else:
                return

            # Standardized queries
            # Metadata-queries
            if variable == "path_meta" and operator1(obj.path_meta, value):
                return obj
            elif variable == "output_frequency_type" and operator1(
                    obj.output_frequency_type, value
            ):
                return obj
            elif variable == "output_frequency" and operator1(
                    obj.output_frequency, value
            ):
                return obj
            elif variable == "general_sigma" and operator1(obj.general_sigma, value):
                return obj
            elif variable == "number_of_circles" and operator1(
                    obj.number_of_circles, value
            ):
                return obj
            elif variable == "number_of_rectangles" and operator1(
                    obj.number_of_rectangles, value
            ):
                return obj
            elif variable == "number_of_runners" and operator1(
                    obj.number_of_runners, value
            ):
                return obj
            elif variable == "number_of_shapes" and operator1(
                    obj.number_of_shapes, value
            ):
                return obj
            # Result-queries
            elif variable == "path_result" and operator1(obj.path_result, value):
                return obj
            elif variable == "avg_level" and operator1(obj.avg_level, value):
                return obj
            elif variable == "age":
                temp = re.search(
                    r"([0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{2}-[0-9]{2}-[0-9]{2})", value
                )
                if temp is not None and operator1(
                        obj.age, datetime.strptime(temp.group(1), "%Y-%m-%d_%H-%M-%S")
                ):
                    return obj
            elif variable == "number_of_sensors" and operator1(
                    obj.number_of_sensors, value
            ):
                return obj

            # Non-standardizable queries (=)
            # Meta-queries
            print(obj.posx_circle)
            if comparisonOperator == "=":
                if (
                        variable == "fibre_content_circles"
                        and np.amin(obj.fibre_content_circles) <= value
                        and np.amax(obj.fibre_content_circles) >= value
                ):
                    return obj
                elif (
                        variable == "fibre_content_rectangles"
                        and np.amin(obj.fibre_content_rectangles) <= value
                        and np.amax(obj.fibre_content_rectangles) >= value
                ):
                    return obj
                elif (
                        variable == "fibre_content_runners"
                        and np.amin(obj.fibre_content_runners) <= value
                        and np.amax(obj.fibre_content_runners) >= value
                ):
                    return obj
                # Circle
                elif variable == "fvc_circle" and operator2(obj.fvc_circle[:], value):
                    return obj
                elif variable == "radius_circle" and operator2(
                        obj.radius_circle, value
                ):
                    return obj
                elif variable == "posx_circle" and operator2(obj.posx_circle[:], value):
                    return obj
                elif variable == "posy_circle" and operator2(obj.posy_circle[:], value):
                    return obj
                # Rectangle
                elif variable == "fvc_rectangle" and operator2(
                        obj.fvc_rectangle[:], value
                ):
                    return obj
                elif variable == "height_rectangle" and operator2(
                        obj.height_rectangle[:], value
                ):
                    return obj
                elif variable == "width_rectangle" and operator2(
                        obj.width_rectangle[:], value
                ):
                    return obj
                elif variable == "posx_rectangle" and operator2(
                        obj.posx_rectangle[:], value
                ):
                    return obj
                elif variable == "posy_rectangle" and operator2(
                        obj.posy_rectangle[:], value
                ):
                    return obj
                # Runner
                elif variable == "fvc_runner" and operator2(obj.fvc_runner[:], value):
                    return obj
                elif variable == "height_runner" and operator2(
                        obj.height_runner[:], value
                ):
                    return obj
                elif variable == "width_runner" and operator2(
                        obj.width_runner[:], value
                ):
                    return obj
                elif variable == "posx_runner" and operator2(obj.posx_runner[:], value):
                    return obj
                elif variable == "posy_runner" and operator2(obj.posy_runner[:], value):
                    return obj
                elif variable == "pos_lower_leftx_runner" and operator2(
                        obj.pos_lower_leftx_runner[:], value
                ):
                    return obj
                elif variable == "pos_lower_lefty_runner" and operator2(
                        obj.pos_lower_lefty_runner[:], value
                ):
                    return obj

            # Standardized queries for > and <
            elif comparisonOperator == ">" or comparisonOperator == "<":
                if variable == "fibre_content_circles" and operator1(
                        operator2(obj.fibre_content_circles), value
                ):
                    return obj
                elif variable == "fibre_content_rectangles" and operator1(
                        operator2(obj.fibre_content_rectangles), value
                ):
                    return obj
                elif variable == "fibre_content_runners" and operator1(
                        operator2(obj.fibre_content_runners), value
                ):
                    return obj
                # Circle
                elif variable == "fvc_circle" and operator1(
                        operator2(obj.fvc_circle), value
                ):
                    return obj
                elif variable == "radius_circle" and operator1(
                        operator2(obj.radius_circle), value
                ):
                    return obj
                elif variable == "posx_circle" and operator1(
                        operator2(obj.posx_circle), value
                ):
                    return obj
                elif variable == "posy_circle" and operator1(
                        operator2(obj.posy_circle), value
                ):
                    return obj
                # Rectangle
                elif variable == "fvc_rectangle" and operator1(
                        operator2(obj.fvc_rectangle), value
                ):
                    return obj
                elif variable == "height_rectangle" and operator1(
                        operator2(obj.height_rectangle), value
                ):
                    return obj
                elif variable == "width_rectangle" and operator1(
                        operator2(obj.width_rectangle), value
                ):
                    return obj
                elif variable == "posx_rectangle" and operator1(
                        operator2(obj.posx_rectangle), value
                ):
                    return obj
                elif variable == "posy_rectangle" and operator1(
                        operator2(obj.posy_rectangle), value
                ):
                    return obj
                # Runner
                elif variable == "fvc_runner" and operator1(
                        operator2(obj.fvc_runner), value
                ):
                    return obj
                elif variable == "height_runner" and operator1(
                        operator2(obj.height_runner), value
                ):
                    return obj
                elif variable == "width_runner" and operator1(
                        operator2(obj.width_runner), value
                ):
                    return obj
                elif variable == "posx_runner" and operator1(
                        operator2(obj.posx_runner), value
                ):
                    return obj
                elif variable == "posy_runner" and operator1(
                        operator2(obj.posy_runner), value
                ):
                    return obj
                elif variable == "pos_lower_leftx_runner" and operator1(
                        operator2(obj.pos_lower_leftx_runner), value
                ):
                    return obj
                elif variable == "pos_lower_lefty_runner" and operator1(
                        operator2(obj.pos_lower_lefty_runner), value
                ):
                    return obj

        return None

    def show_selection_options(self):
        self.options = PrettyTable()
        self.options.field_names = ["Possible options are"]
        # Metadata
        self.options.add_row(["path_meta"])
        self.options.add_row(["output_frequency_type"])
        self.options.add_row(["output_frequency"])
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
        self.options.add_row(["posy_circle"])
        self.options.add_row(["fvc_rectangle"])
        self.options.add_row(["height_rectangle"])
        self.options.add_row(["width_rectangle"])
        self.options.add_row(["posx_rectangle"])
        self.options.add_row(["posy_rectangle"])
        self.options.add_row(["fvc_runner"])
        self.options.add_row(["height_runner"])
        self.options.add_row(["width_runner"])
        self.options.add_row(["posx_runner"])
        self.options.add_row(["posy_runner"])
        self.options.add_row(["pos_lower_leftx_runner"])
        self.options.add_row(["pos_lower_lefty_runner"])
        # Result
        self.options.add_row(["path_result"])
        self.options.add_row(["avg_level"])
        self.options.add_row(["age"])
        self.options.add_row(["number_of_sensors"])
        print(self.options)

    def show_objects(self):
        if not (len(self.hdf5_object_list) == 0):
            for val in self.hdf5_object_list:
                val.show_object_content()
                print("\n")
        else:
            print("No objects were found.")

    def get_meta_path_list(self):
        self.meta = [obj.path_meta for obj in self.hdf5_object_list]
        return self.meta

    def get_result_path_list(self):
        self.result = [obj.path_result for obj in self.hdf5_object_list]
        return self.result

    def save(self, path, filename="HDF5DB"):
        dirpath = Path(path)
        if dirpath.is_dir():
            if not (len(self.hdf5_object_list) == 0):
                file = Path(filename + r".h5db")
                h5db_path = dirpath / file
                if h5db_path.is_file():
                    print(
                        "A file with the given name already exists. "
                        + filename
                        + " will be overwritten. Do you want to continue?\nPlease type in yes to continue."
                    )
                    userinput = input("")
                    if not (userinput == "yes"):
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
        if h5db_path.is_file():
            infile = open(dir_path / Path(filename + ".h5db"), "rb")
            self.hdf5_object_list = pickle.load(infile)
            self.meta = [obj.path_meta for obj in self.hdf5_object_list]
            self.result = [obj.path_result for obj in self.hdf5_object_list]
            infile.close()
            print("HDF5DB loaded")
        else:
            print("There is no h5db-file at the given path with the given name!")
