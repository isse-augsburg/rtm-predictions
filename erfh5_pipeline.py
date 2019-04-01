import h5py 
import matplotlib.pyplot as plt 
import matplotlib
import os 
import time 
from multiprocessing import Process, Queue
from functools import partial
from os import listdir, walk 
import queue 
import threading 
import random 
from enum import Enum


class Queue_State(Enum):
    empty = 0
    loading = 1
    complete = 2
    uninitialized = 3

class Batched_Queue(): 

    def __init__(self):
        self.queue = Queue()

    def put_batch(self, items):
        for i in items:
            self.queue.put(i)

    def get_batch(self, number_of_items):
        result = []
        for i in range(number_of_items): 
            result.append(self.queue.get(block=True))
        return result 

class Thread_Safe_List():
    
    def __init__(self, max_length=-1):
        self.list = []
        self.lock = threading.Lock()
        self.max_length = max_length
        self.should_terminate = False
    
    # if called, the thread using this queue will commit suicide
    def kill(self):
        self.should_terminate = True

    def put(self, element): 

        while len(self.list) >= self.max_length and self.max_length != -1: 
            if self.should_terminate:
                raise Exception("Oof")
            time.sleep(0.1)
        
        self.lock.acquire()
        self.list.append(element)
        
        self.lock.release()
    
    def put_batch(self, batch): 

        self.lock.acquire()
        self.list.extend(batch)
        self.lock.release()

    def get(self, number_of_elements): 
        
        while len(self.list) < number_of_elements: 
            if self.should_terminate:
                raise Exception("Big oof")
            time.sleep(0.1)
        
        self.lock.acquire()
        items = self.list[0:number_of_elements]
        self.list = self.list[number_of_elements:]
        self.lock.release()
        return items 



class ERFH5_DataGenerator():
    
    def __init__(self, data_path='/home/', batch_size=64, repeat=True, epochs=80):
        self.data_path = data_path
        self.batch_size = batch_size
        self.paths = self.__get_paths_to_files(self.data_path)
        self.path_queue = Thread_Safe_List()
        self.path_queue.put_batch(self.paths)
        self.num_workers = 6
        self.epochs = epochs

        self._queueState = Queue_State.uninitialized
        
        self.batch_queue = Thread_Safe_List()
        #self.batch_queue = Queue()
        self.threads = []
        t_path = threading.Thread(target=self.__fill_path_queue)
        t_path.start()

        for i in range(self.num_workers):
            t_batch = threading.Thread(target=self.__fill_batch_queue)
            t_batch.start()
            self.threads.append(t_batch)
            
        

    def __get_paths_to_files(self, data_path):
        dataset_filenames = []
        for (dirpath, dirnames, filenames) in walk(data_path):
            if filenames: 
                filenames = [dirpath + '/' + f for f in filenames]
                dataset_filenames.extend(filenames)
        return dataset_filenames   

    def __fill_batch_queue(self):
        while self._queueState == Queue_State.uninitialized:
            time.sleep(1)


        while not self._queueState == Queue_State.empty:
            
            file = self.path_queue.get(1)[0]
            if(len(self.path_queue.list) == 0 and self._queueState == Queue_State.complete):
                self._queueState = Queue_State.empty
                self.batch_queue.kill()
            data = self.__create_filling_factors_dataset(file)
            self.batch_queue.put(data)
    
    def __fill_path_queue(self): 
        self._queueState = Queue_State.loading
        for i in range(self.epochs):
            new_paths = self.paths
            random.shuffle(new_paths)
            self.path_queue.put_batch(new_paths)
        self._queueState = Queue_State.complete
   
    
    def __create_filling_factors_dataset(self, filenames): 
           
        states_and_fillings = self.__get_states_and_fillings(filenames)
        label = int(states_and_fillings[-1][0])
        states_and_fillings = [i[1] for i in states_and_fillings]
        instance = (states_and_fillings, label)
        return instance
    
    
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
        print("Batch")
        start_time = time.time()
        try:
            batch = self.batch_queue.get(self.batch_size)
        except Exception as e:
            raise e 
        
        duration = time.time() - start_time
        print(duration)

        data = [i[0]for i in batch]
        labels = [i[1] for i in batch]
        
        #return batch[:,0], batch[:,1]
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


if __name__== "__main__":
    data_folder = '/home/lodes/Sim_Results'
    generator = ERFH5_DataGenerator(data_path=data_folder, batch_size=1, epochs=1)
    try:
        batch_data, batch_labels = generator.get_batch()
    except Exception as e:
        print("oopsie")
    
    while True:
        print("foo")
        time.sleep(2)