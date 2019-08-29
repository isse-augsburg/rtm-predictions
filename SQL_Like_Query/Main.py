from HDF5DB import HDF5DB
from HDF5Object import HDF5Object

def main():
  test = HDF5DB()
  #test2 = HDF5Object("Data/0") #/2019-07-23_15-38-08_0
  #test.addObject(test2)
  test.addObjectsFromPath("Data")
  #test.showObjects()
  test.save("Data", "HDF5DB")
  test.load("Data", "HDF5DB")
  test.showSelectionOptions()
  #test.showObjects()
  #test.select("age", "2019-07-23_15-38-05", ">")
  #test.select("outputFrequencyType", 1, ">")
  #test.select("fibreContentRunners", -0.7, "=")
  test.select("widthRectangle", 6, "<")
  test.showObjects()
  
if __name__== "__main__":
    main()

#ToDo:
#       Testklasse für HDF5DB sowie HDF5Object
#       Comments
#       Add x,y
#       Check paths in erfh5 and hdf5