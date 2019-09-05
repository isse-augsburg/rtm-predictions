   def select2(self, variable, value, comparisonOperator="="):
        self.selected = []
        for obj in self.hdf5_object_list:
            if comparisonOperator == "=" and hasattr(obj, variable):
                if variable == "path_meta" and obj.path_meta == value:
                    self.selected.append(obj)
                elif (
                    variable == "output_frequency_type"
                    and obj.output_frequency_type == value
                ):
                    self.selected.append(obj)
                elif variable == "output_frequency" and obj.output_frequency == value:
                    self.selected.append(obj)
                elif variable == "general_sigma" and obj.general_Sigma == value:
                    self.selected.append(obj)
                elif variable == "number_of_circles" and obj.number_of_circles == value:
                    self.selected.append(obj)
                elif (
                    variable == "number_of_rectangles"
                    and obj.number_of_rectangles == value
                ):
                    self.selected.append(obj)
                elif variable == "number_of_runners" and obj.number_of_runners == value:
                    self.selected.append(obj)
                elif variable == "number_of_shapes" and obj.number_of_shapes == value:
                    self.selected.append(obj)
                elif (
                    variable == "fibre_content_circles"
                    and np.amin(obj.fibre_content_circles) <= value
                    and np.amax(obj.fibre_content_circles) >= value
                ):
                    self.selected.append(obj)
                elif (
                    variable == "fibre_content_rectangles"
                    and np.amin(obj.fibre_content_rectangles) <= value
                    and np.amax(obj.fibre_content_rectangles) >= value
                ):
                    self.selected.append(obj)
                elif (
                    variable == "fibre_content_runners"
                    and np.amin(obj.fibre_content_runners) <= value
                    and np.amax(obj.fibre_content_runners) >= value
                ):
                    self.selected.append(obj)
                # Circle
                elif variable == "fvc_circle" and value in obj.fvc_circle:
                    self.selected.append(obj)
                elif variable == "radius_circle" and value in obj.radius_circle:
                    self.selected.append(obj)
                elif variable == "pox_circle" and value in obj.posx_circle:
                    self.selected.append(obj)
                elif variable == "posy_circle" and value in obj.posy_circle:
                    self.selected.append(obj)
                # Rectangle
                elif variable == "fvc_rectangle" and value in obj.fvc_rectangle:
                    self.selected.append(obj)
                elif variable == "height_rectangle" and value in obj.height_rectangle:
                    self.selected.append(obj)
                elif variable == "width_rectangle" and value in obj.width_rectangle:
                    self.selected.append(obj)
                elif variable == "posx_rectangle" and value in obj.posx_rectangle:
                    self.selected.append(obj)
                elif variable == "posy_rectangle" and value in obj.posy_rectangle:
                    self.selected.append(obj)
                # Runner
                elif variable == "fvc_runner" and value in obj.fvc_runner:
                    self.selected.append(obj)
                elif variable == "height_runner" and value in obj.height_runner:
                    self.selected.append(obj)
                elif variable == "width_runner" and value in obj.width_runner:
                    self.selected.append(obj)
                elif variable == "posx_runner" and value in obj.posx_runner:
                    self.selected.append(obj)
                elif variable == "posy_runner" and value in obj.posy_runner:
                    self.selected.append(obj)
                elif (
                    variable == "pos_lower_left__runner"
                    and value in obj.pos_lower_leftx_runner
                ):
                    self.selected.append(obj)
                elif (
                    variable == "pos_lower_lefty_runner"
                    and value in obj.pos_lower_lefty_runner
                ):
                    self.selected.append(obj)
                # Result
                elif variable == "path_result" and obj.path_result == value:
                    self.selected.append(obj)
                elif variable == "avg_level" and obj.avg_level == value:
                    self.selected.append(obj)
                elif variable == "age" and obj.age == datetime.strptime(
                    re.search(
                        "([0-9]{4}\-[0-9]{2}\-[0-9]{2}_[0-9]{2}\-[0-9]{2}\-[0-9]{2})",
                        value,
                    ).group(1),
                    "%Y-%m-%d_%H-%M-%S",
                ):
                    self.selected.append(obj)
                elif variable == "number_of_sensors" and obj.number_of_sensors == value:
                    self.selected.append(obj)

            elif comparisonOperator == ">" and hasattr(obj, variable):
                if variable == "path_meta":
                    print(
                        "The operator "
                        + comparisonOperator
                        + " is not available for metaPath."
                    )
                    return
                elif (
                    variable == "output_frequency_type"
                    and obj.output_frequency_type > value
                ):
                    self.selected.append(obj)
                elif variable == "output_frequency" and obj.output_frequency > value:
                    self.selected.append(obj)
                elif variable == "general_sigma" and obj.general_sigma > value:
                    self.selected.append(obj)
                elif variable == "number_of_circles" and obj.number_of_circles > value:
                    self.selected.append(obj)
                elif (
                    variable == "number_of_rectangles"
                    and obj.number_of_rectangles > value
                ):
                    self.selected.append(obj)
                elif variable == "number_of_runners" and obj.number_of_runners > value:
                    self.selected.append(obj)
                elif variable == "number_of_shapes" and obj.number_of_shapes > value:
                    self.selected.append(obj)
                elif (
                    variable == "fibre_content_circles"
                    and np.amin(obj.fibre_content_circles) > value
                ):
                    self.selected.append(obj)
                elif (
                    variable == "fibre_content_rectangles"
                    and np.amin(obj.fibre_content_rectangles) > value
                ):
                    self.selected.append(obj)
                elif (
                    variable == "fibre_content_runners"
                    and np.amin(obj.fibre_content_runners) > value
                ):
                    self.selected.append(obj)
                # Circle
                elif variable == "fvc_circle" and np.amin(obj.fvc_circle) > value:
                    self.selected.append(obj)
                elif variable == "radius_circle" and np.amin(obj.radius_circle) > value:
                    self.selected.append(obj)
                elif variable == "posx_circle" and np.amin(obj.posx_circle) > value:
                    self.selected.append(obj)
                elif variable == "posy_circle" and np.amin(obj.posy_circle) > value:
                    self.selected.append(obj)
                # Rectangle
                elif variable == "fvc_rectangle" and np.amin(obj.fvc_rectangle) > value:
                    self.selected.append(obj)
                elif (
                    variable == "height_rectangle"
                    and np.amin(obj.height_rectangle) > value
                ):
                    self.selected.append(obj)
                elif (
                    variable == "width_rectangle"
                    and np.amin(obj.width_rectangle) > value
                ):
                    self.selected.append(obj)
                elif (
                    variable == "posx_rectangle" and np.amin(obj.posx_rectangle) > value
                ):
                    self.selected.append(obj)
                elif (
                    variable == "posy_rectangle" and np.amin(obj.posy_rectangle) > value
                ):
                    self.selected.append(obj)
                # Runner
                elif variable == "fvc_runner" and np.amin(obj.fvc_runner) > value:
                    self.selected.append(obj)
                elif variable == "height_runner" and np.amin(obj.height_runner) > value:
                    self.selected.append(obj)
                elif variable == "width_runner" and np.amin(obj.width_runner) > value:
                    self.selected.append(obj)
                elif variable == "posx_runner" and np.amin(obj.posx_runner) > value:
                    self.selected.append(obj)
                elif variable == "posy_runner" and np.amin(obj.posy_runner) > value:
                    self.selected.append(obj)
                elif (
                    variable == "pos_lower_leftx_runner"
                    and np.amin(obj.pos_lower_leftx_runner) > value
                ):
                    self.selected.append(obj)
                elif (
                    variable == "pos_lower_lefty_runner"
                    and np.amin(obj.pos_lower_lefty_runner) > value
                ):
                    self.selected.append(obj)
                # Result
                elif variable == "path_result":
                    print(
                        "The operator "
                        + comparisonOperator
                        + " is not available for path_result."
                    )
                    return
                elif variable == "avg_level" and obj.avg_level > value:
                    self.selected.append(obj)
                elif variable == "age" and obj.age > datetime.strptime(
                    re.search(
                        "([0-9]{4}\-[0-9]{2}\-[0-9]{2}_[0-9]{2}\-[0-9]{2}\-[0-9]{2})",
                        value,
                    ).group(1),
                    "%Y-%m-%d_%H-%M-%S",
                ):
                    self.selected.append(obj)
                elif variable == "number_of_sensors" and obj.number_of_sensors > value:
                    self.selected.append(obj)

            elif comparisonOperator == "<" and hasattr(obj, variable):
                if variable == "path_meta":
                    print(
                        "The operator "
                        + comparisonOperator
                        + " is not available for path_meta."
                    )
                    return
                elif (
                    variable == "output_frequency_type"
                    and obj.output_frequency_type < value
                ):
                    self.selected.append(obj)
                elif variable == "output_frequency" and obj.output_frequency < value:
                    self.selected.append(obj)
                elif variable == "general_sigma" and obj.general_sigma < value:
                    self.selected.append(obj)
                elif variable == "number_of_circles" and obj.number_of_circles < value:
                    self.selected.append(obj)
                elif (
                    variable == "number_of_rectangles"
                    and obj.number_of_rectangles < value
                ):
                    self.selected.append(obj)
                elif variable == "number_of_runners" and obj.number_of_runners < value:
                    self.selected.append(obj)
                elif variable == "number_of_shapes" and obj.number_of_shapes < value:
                    self.selected.append(obj)
                elif (
                    variable == "fibre_content_circles"
                    and np.amax(obj.fibre_content_circles) < value
                ):
                    self.selected.append(obj)
                elif (
                    variable == "fibre_content_rectangles"
                    and np.amax(obj.fibre_content_rectangles) < value
                ):
                    self.selected.append(obj)
                elif (
                    variable == "fibre_content_runners"
                    and np.amax(obj.fibre_content_runners) < value
                ):
                    self.selected.append(obj)
                # Circle
                elif variable == "fvc_circle" and np.amax(obj.fvc_circle) < value:
                    self.selected.append(obj)
                elif variable == "radius_circle" and np.amax(obj.radius_circle) < value:
                    self.selected.append(obj)
                elif variable == "posx_circle" and np.amax(obj.posx_circle) < value:
                    self.selected.append(obj)
                elif variable == "posy_circle" and np.amax(obj.posy_circle) < value:
                    self.selected.append(obj)
                # Rectangle
                elif variable == "fvc_rectangle" and np.amax(obj.fvc_rectangle) < value:
                    self.selected.append(obj)
                elif (
                    variable == "height_rectangle"
                    and np.amax(obj.height_rectangle) < value
                ):
                    self.selected.append(obj)
                elif (
                    variable == "width_rectangle"
                    and np.amax(obj.width_rectangle) < value
                ):
                    self.selected.append(obj)
                elif (
                    variable == "posx_rectangle" and np.amax(obj.posx_rectangle) < value
                ):
                    self.selected.append(obj)
                elif (
                    variable == "posy_rectangle" and np.amax(obj.posy_rectangle) < value
                ):
                    self.selected.append(obj)
                # Runner
                elif variable == "fvc_runner" and np.amax(obj.fvc_runner) < value:
                    self.selected.append(obj)
                elif variable == "height_runner" and np.amax(obj.height_runner) < value:
                    self.selected.append(obj)
                elif variable == "width_runner" and np.amax(obj.width_runner) < value:
                    self.selected.append(obj)
                elif variable == "posx_runner" and np.amax(obj.posx_runner) < value:
                    self.selected.append(obj)
                elif variable == "posy_runner" and np.amax(obj.posy_runner) < value:
                    self.selected.append(obj)
                elif (
                    variable == "pos_lower_leftx_runner"
                    and np.amax(obj.pos_lower_leftx_runner) < value
                ):
                    self.selected.append(obj)
                elif (
                    variable == "pos_lower_lefty_runner"
                    and np.amax(obj.pos_lower_lefty_runner) < value
                ):
                    self.selected.append(obj)
                # Result
                elif variable == "path_result":
                    print(
                        "The operator "
                        + comparisonOperator
                        + " is not available for path_result."
                    )
                    return
                elif variable == "avg_level" and obj.avg_level < value:
                    self.selected.append(obj)
                elif variable == "age" and obj.age < datetime.strptime(
                    re.search(
                        "([0-9]{4}\-[0-9]{2}\-[0-9]{2}_[0-9]{2}\-[0-9]{2}\-[0-9]{2})",
                        value,
                    ).group(1),
                    "%Y-%m-%d_%H-%M-%S",
                ):
                    self.selected.append(obj)
                elif variable == "number_of_sensors" and obj.number_of_sensors < value:
                    self.selected.append(obj)

        if len(self.selected) == 0:
            print(
                "No matches were found for "
                + str(variable)
                + " "
                + str(comparisonOperator)
                + " "
                + str(value)
                + ". No filter was applied!"
            )
        else:
            self.HDF5Object = self.selected
            if len(self.selected) > 1:
                print(
                    "\nThe filter "
                    + str(variable)
                    + " "
                    + str(comparisonOperator)
                    + " "
                    + str(value)
                    + " was applied. "
                    + str(len(self.selected))
                    + " objects were found."
                )
                self.meta = [obj.path_meta for obj in self.hdf5_object_list]
                self.result = [obj.path_result for obj in self.hdf5_object_list]
            else:
                print(
                    "\nThe filter "
                    + str(variable)
                    + " "
                    + str(comparisonOperator)
                    + " "
                    + str(value)
                    + " was applied. "
                    + str(len(self.selected))
                    + " object was found."
                )
            self.hdf5_object_list = self.selected
