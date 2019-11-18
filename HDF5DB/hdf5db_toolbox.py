import operator
import pickle
import re
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np
from prettytable import PrettyTable
from tqdm import tqdm

from HDF5DB.hdf5db_object import HDF5Object


class HDF5DBToolbox:
    def __init__(self):
        self.hdf5_object_list = []

    def add_object(self, object):
        if object is not None:
            self.hdf5_object_list.append(object)

    def add_objects_from_path(self, path):
        self.get_meta_path_list()
        temp = len(self.hdf5_object_list)
        erfh5_path = []
        hdf5_path = []
        dirpath = Path(path)
        if dirpath.is_dir():
            # List all hdf5-files
            print("Data is being retrieved...")
            for i in tqdm([el for el in tqdm(dirpath.rglob("**/*.hdf5"))]):
                # Check that only *.hdf5 and *.erfh5 files will be opened
                if h5py.is_hdf5(i.as_posix()):
                    erfh5File = Path(str(i).replace("meta_data.hdf5", "RESULT.erfh5"))
                    if erfh5File.exists():
                        hdf5_path.append(i.as_posix())
                        erfh5_path.append(erfh5File.as_posix())
                else:
                    print(f"{str(i)} does not exist. The folder was skipped.")
            print("H5-files are currently being scanned...")
            with Pool(20) as p:
                new_objects = p.starmap(
                    HDF5Object, zip(hdf5_path, erfh5_path)
                )
            print("Objects are currently being added...")
            for i in new_objects:
                if i.path_meta not in self.get_meta_path_list():
                    self.hdf5_object_list.append(i)
            print(
                f"{str(abs(temp - len(self.hdf5_object_list)))} Objects have been added."
            )
        else:
            print(f"The path {path} does not exist! No objects were added!")

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
        selected = [a for a in selected if a is not None]

        if len(selected) == 0:
            print(
                f"No matches were found for {variable} {comparisonOperator} {value}. No filter was applied!"
                + " Maybe the operator is not available for this parameter."
            )
            return -1
        else:
            self.hdf5_object_list = selected
            if len(selected) > 1:
                print(
                    f"The filter {variable} {operator} {value} was applied. {len(selected)} objects were found."
                )
            else:
                print(
                    f"The filter {variable} {operator} {value} was applied. {len(selected)} object was found."
                )
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
            if variable == "path_meta" and comparisonOperator == "=":
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
            elif variable == "path_result" and comparisonOperator == "=":
                return obj
            elif variable == "avg_level" and operator1(obj.avg_level, value):
                return obj
            elif variable == "single_states" and operator1(obj.single_states, value):
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
            if comparisonOperator == "=":
                if (
                        variable == "fibre_content_circles"
                        and np.amin(obj.fibre_content_circles) <= value <= np.amax(obj.fibre_content_circles)
                ):
                    return obj
                elif (
                        variable == "fibre_content_rectangles"
                        and np.amin(obj.fibre_content_rectangles) <= value <= np.amax(obj.fibre_content_rectangles)
                ):
                    return obj
                elif (
                        variable == "fibre_content_runners"
                        and np.amin(obj.fibre_content_runners) <= value <= np.amax(obj.fibre_content_runners)
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
        options = PrettyTable()
        options.field_names = ["Possible options are"]
        # Metadata
        options.add_row(["path_meta"])
        options.add_row(["output_frequency_type"])
        options.add_row(["output_frequency"])
        options.add_row(["general_sigma"])
        options.add_row(["number_of_circles"])
        options.add_row(["number_of_rectangles"])
        options.add_row(["number_of_runners"])
        options.add_row(["number_of_shapes"])
        options.add_row(["fibre_content_circles"])
        options.add_row(["fibre_content_rectangles"])
        options.add_row(["fibre_content_runners"])
        options.add_row(["fvc_circle"])
        options.add_row(["radius_circle"])
        options.add_row(["posx_circle"])
        options.add_row(["posy_circle"])
        options.add_row(["fvc_rectangle"])
        options.add_row(["height_rectangle"])
        options.add_row(["width_rectangle"])
        options.add_row(["posx_rectangle"])
        options.add_row(["posy_rectangle"])
        options.add_row(["fvc_runner"])
        options.add_row(["height_runner"])
        options.add_row(["width_runner"])
        options.add_row(["posx_runner"])
        options.add_row(["posy_runner"])
        options.add_row(["pos_lower_leftx_runner"])
        options.add_row(["pos_lower_lefty_runner"])
        # Result
        options.add_row(["path_result"])
        options.add_row(["avg_level"])
        options.add_row(["single_states"])
        options.add_row(["age"])
        options.add_row(["number_of_sensors"])
        print(options)

    def show_objects(self):
        if not (len(self.hdf5_object_list) == 0):
            for val in self.hdf5_object_list:
                val.show_object_content()
                print("\n")
        else:
            print("No objects were found.")

    def get_meta_path_list(self):
        meta = [obj.path_meta for obj in self.hdf5_object_list]
        return meta

    def get_result_path_list(self):
        result = [obj.path_result for obj in self.hdf5_object_list]
        return result

    def save(self, path, filename="HDF5DB"):
        dirpath = Path(path)
        if dirpath.is_dir():
            if not (len(self.hdf5_object_list) == 0):
                file = Path(filename + r".hdf5db")
                h5db_path = dirpath / file
                if h5db_path.is_file():
                    print(
                        f"A file with the given name already exists. {filename} will be overwritten. "
                        "Do you want to continue?\nPlease type in yes to continue."
                    )
                    userinput = input("")
                    if not (userinput == "yes"):
                        print("Nothing has been saved.")
                        return
                    else:
                        print(f"{filename} will be overwritten.")
                outfile = open(dirpath / file, "wb")
                pickle.dump(self.hdf5_object_list, outfile)
                outfile.close()
                print("HDF5DB saved")
            else:
                print("No objects were found. Nothing was saved!")
        else:
            print(f"{path} does not exist! Nothing was saved!")

    def force_save(self, path, filename="HDF5DB"):
        dirpath = Path(path)
        if dirpath.is_dir():
            if not (len(self.hdf5_object_list) == 0):
                file = Path(filename + r".hdf5db")
                h5db_path = dirpath / file
                outfile = open(h5db_path, "wb")
                pickle.dump(self.hdf5_object_list, outfile)
                outfile.close()
                print("HDF5DB saved")
            else:
                print("No objects were found. Nothing was saved!")
        else:
            print(f"{path} does not exist! Nothing was saved!")

    def load(self, path, filename="HDF5DB"):
        dir_path = Path(path)
        h5db_path = dir_path / Path(filename + str(".h5db"))
        if h5db_path.is_file():
            infile = open(dir_path / Path(filename + ".h5db"), "rb")
            self.hdf5_object_list = pickle.load(infile)
            infile.close()
            print("HDF5DB loaded")
        else:
            print("There is no h5db-file at the given path with the given name!")
