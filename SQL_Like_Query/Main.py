from HDF5DB import HDF5DB


def main():
  test = HDF5DB()
  # test2 = HDF5Object("Data/0") #/2019-07-23_15-38-08_0
  # test.addObject(test2)
  test.add_objects_from_path("Data")
  # test.showObjects()
  test.save("Data", "HDF5DB")
  test.load("Data", "HDF5DB")
  test.show_selection_options()
  # test.showObjects()
  # test.select("age", "2019-07-23_15-38-05", ">")
  # test.select("output_frequency_type", 1, ">")
  # test.select("fibre_content_runners", -0.7, "=")
  test.select("avg_level", 0.8, ">")
  test.show_objects()

if(__name__ == "__main__"):
    main()

# TODO:
#       Testklasse für HDF5DB sowie HDF5Object
#       outputfrequency type speichern, evtl. in result, ansonsten simcreator ändern