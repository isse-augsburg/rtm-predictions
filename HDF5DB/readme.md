****Short guide for the usage of the HDF5DB****    

First of all you have to import the HDF5DB-Toolbox via:  
	from HDF5DB.hdf5db_toolbox import HDF5DBToolbox  
Afterwords create a HDF5Toolbox-instance to address the different functions.  
	e. g. hdf5db = HDF5Toolbox()  
After these steps, you can execute the following commands:  
        1. Add:  
    To add data from a Simulation, please use the add_objects_from_path(self, path)-method.  
   	e. g. hdf5db.add_objects_from_path(/cfs/home/s/t/stiebesi/data/RTM/Leoben/output/with_shapes/  
	2019-07-23_15-38-08_5000p"  
    2. Select:  
   To select a part of the simulation data, please use the select(self, variable,  
   comparisonOperator="=", value)-method.  
	e. g. hdf5db.select("fvc_circle", ">", 0.2)  
    3. Show selection options:  
    To get the different possible selection options for the selectable variables displayed,  
    please use the show_selection_options(self)-method.  
	e. g. hdf5db.show_selection_options()  
    4. Show objects:  
    In order to view the selected simulation data please use the show_objects(self)-method.  
	e. g. hdf5db.show_objects()  
    5. Get meta-paths:  
    To get the list of metadata-paths of the currently selected simulations, please use get_meta_path_list(self)-method.  
	e. g. hdf5db.get_meta_path_list()  
    6. Get result-paths:  
    To get the list of resultdata-paths of the currently selected simulations, please use get_result_path_list(self)  
	e. g. hdf5db.get_result_path_list()  
    7. Save:  
    To save the current selection as a file, please use the save(self, path, filename="HDF5DB")-method.  
	e. g. hdf5db.save("/cfs/share/cache/HDF5DB_Cache",  "Test")  
    8. Load:  
    To load a selection from a file please use the load(self, path, filename="HDF5DB")-method.  
	e. g. hdf5db..load("/cfs/share/cache/HDF5DB_Cache", "Test")
