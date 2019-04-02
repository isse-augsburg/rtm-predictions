import h5py
from os import listdir, walk 
import time

def __get_states_and_fillings(filename):
    t1 = time.time()
    f = h5py.File(filename, 'r')
    coord_as_np_array = f['post/constant/entityresults/NODE/COORDINATE/ZONE1_set0/erfblock/res'].value
    # Cut off last column (z), since it is filled with 1s anyway
    _coords = coord_as_np_array[:, :-1]
    all_states = f['post']['singlestate']
    
    filling_factors_at_certain_times = [f['post']['singlestate'][state]['entityresults']['NODE']['FILLING_FACTOR']['ZONE1_set1']['erfblock']['res'][()] for state in all_states]
         
    states_as_list = [x[-5:] for x in list(all_states.keys())]
    flat_fillings = [x.flatten() for x in filling_factors_at_certain_times]
    states_and_fillings = [(i, j) for i, j in zip(states_as_list, flat_fillings)]
    print(time.time() -t1)  
    return states_and_fillings


def __get_paths_to_files(root_directory):
        dataset_filenames = []
        for (dirpath, dirnames, filenames) in walk(root_directory):
            if filenames: 
                filenames = [dirpath + '/' + f for f in filenames]
                dataset_filenames.extend(filenames)
        return dataset_filenames   


if __name__== "__main__":
     data_folder = '/home/niklas/Documents/Data'
     files = __get_paths_to_files(data_folder)
     for f in files:
         __get_states_and_fillings(f)
