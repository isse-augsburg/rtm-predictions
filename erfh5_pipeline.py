import h5py 
import matplotlib.pyplot as plt 
import matplotlib
import os 
import time 
from multiprocessing import Pool, Queue, Manager
from functools import partial
from os import listdir, walk 
import queue 


class Batched_Queue(): 

    def __init__(self, max_size):
        self.max_size = max_size 
        self.queue = Queue(maxsize=max_size)

    def put_batch(self, items):
        for i in items:
            self.queue.put(i)

    def get_batch(self, number_of_items):
        result = []
        for i in range(number_of_items): 
            result.append(self.queue.get())
        return result 



class ERFH5_DataGenerator():
    
    def __init__(self, data_path='/home/', batch_size=64, repeat=True):
        self.data_path = data_path
        self.batch_size = batch_size
        self.paths = self.__get_paths_to_files(self.data_path)
        #TODO shuffle 
        #TODO repeat 
        self.batch_queue = Batched_Queue(max_size=1000000)
        self.__fill_batch_queue()

    def __get_paths_to_files(self, data_path):
        dataset_filenames = Batched_Queue(max_size=1000000)
        for (dirpath, dirnames, filenames) in walk(data_folder):
            if filenames: 
                filenames = [dirpath + '/' + f for f in filenames]
                dataset_filenames.put_batch(filenames)
        return dataset_filenames   

    def __fill_batch_queue(self):
            data = self.__create_filling_factors_dataset(self.paths.get_batch(3*self.batch_size))
            self.batch_queue.put_batch(data)
    
    def __create_filling_factors_dataset(self, filenames): 
        dataset = []
        for filename in filenames:    
            states_and_fillings = self.__get_states_and_fillings(filename)
            label = int(states_and_fillings[-1][0])
            instance = [states_and_fillings, label]
            dataset.append(instance)

        return dataset
    
    def __get_states_and_fillings(self, filename):
        f = h5py.File(filename, 'r')

        coord_as_np_array = f['post/constant/entityresults/NODE/COORDINATE/ZONE1_set0/erfblock/res'].value
        # Cut off last column (z), since it is filled with 1s anyway
        _coords = coord_as_np_array[:, :-1]
        all_states = f['post']['singlestate']
        filling_factors_at_certain_times = [f['post']['singlestate'][state]['entityresults']['NODE']['FILLING_FACTOR']['ZONE1_set1']['erfblock']['res'][()] for state in all_states]
        states_as_list = [x[-5:] for x in list(all_states.keys())]
        flat_fillings = [x.flatten() for x in filling_factors_at_certain_times]
        states_and_fillings = [(i, j) for i, j in zip(states_as_list, flat_fillings)]
        return states_and_fillings

    def get_batch(self):
        batch = self.batch_queue.get_batch(self.batch_size)
        data = [i[0]for i in batch]
        labels = [i[1] for i in batch]
        self.__fill_batch_queue()
        return data, labels 
   
########################################################################################
########################################################################################

def get_states_and_fillings(filename):
    f = h5py.File(filename, 'r')

    coord_as_np_array = f['post/constant/entityresults/NODE/COORDINATE/ZONE1_set0/erfblock/res'].value
    # Cut off last column (z), since it is filled with 1s anyway
    _coords = coord_as_np_array[:, :-1]
    all_states = f['post']['singlestate']
    filling_factors_at_certain_times = [f['post']['singlestate'][state]['entityresults']['NODE']['FILLING_FACTOR']['ZONE1_set1']['erfblock']['res'][()] for state in all_states]
    states_as_list = [x[-5:] for x in list(all_states.keys())]
    flat_fillings = [x.flatten() for x in filling_factors_at_certain_times]
    states_and_fillings = [(i, j) for i, j in zip(states_as_list, flat_fillings)]
    return states_and_fillings

def run_erfh5(filename):
   
    states_and_fillings = get_states_and_fillings(filename)

    t0 = time.time()
    with Pool(6) as p:
        res = p.map(partial(plot_wrapper, coords= _coords), states_and_fillings)
    print('Done after', time.time() -t0)


def plot_wrapper(states_and_filling, coords):
    filename = r'C:\Users\stiebesi\code\datamuddler\plots\lautern_flawless_hd\%s.png' % str(states_and_filling[0])
    if os.path.exists(filename):
        return False
    fig = plt.gcf()
    fig.set_size_inches(18.5, 18.5)
    areas = len(states_and_filling[1]) * [3]
    norm = matplotlib.cm.colors.Normalize(vmax=states_and_filling[1].max(), vmin=states_and_filling[1].min())
    plt.scatter(coords[:, 0], coords[:, 1], c=states_and_filling[1], s=areas, norm=norm)
    fig.savefig(filename)
    return True

def create_filling_factors_dataset(filenames): 
    dataset = []
    for filename in filenames:    
        states_and_fillings = get_states_and_fillings(filename)
        label = int(states_and_fillings[-1][0])
        instance = [states_and_fillings, label]
        dataset.append(instance)

    return dataset

######################################################################################################
######################################################################################################
""" data_folder = '/home/lodes/Sim_Results'
dataset_filenames = []
for (dirpath, dirnames, filenames) in walk(data_folder):
    if filenames: 
        filenames = [dirpath + '/' + f for f in filenames]
        dataset_filenames.extend(filenames)


dataset = create_filling_factors_dataset(dataset_filenames)
data = [i[0]for i in dataset]
labels = [i[1] for i in dataset]
print("foo") """

if __name__== "__main__":
    data_folder = '/home/lodes/Sim_Results'
    generator = ERFH5_DataGenerator(data_path=data_folder, batch_size=2)
    batch_data, batch_labels = generator.get_batch()
    print("foo")
