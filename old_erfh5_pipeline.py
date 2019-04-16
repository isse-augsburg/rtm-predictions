import h5py 
import matplotlib.pyplot as plt 
import matplotlib
import os 
import time 
import numpy as np 
import torch
from multiprocessing import Process, Queue
from multiprocessing import cpu_count
from functools import partial
from os import listdir, walk 
import queue 
import threading 
import random 
from enum import Enum


class Pipeline_Mode(Enum):
    index_sequence = 0
    single_instance = 1
    time_sequence = 2

class NoSequenceException(Exception):
    pass 


#Helper data structure that's thread safe and supports batch-wise get 
class Thread_Safe_List():
    
    def __init__(self, max_length=-1):
        self.list = []
        self.lock = threading.Lock()
        self.max_length = max_length
        self.finished = False
    
    # if called, the thread using this queue will commit suicide
    def kill(self):
        self.finished = True


    def __len__(self):
        #self.lock.acquire()
        length = len(self.list)
        #self.lock.release()
        return length

    def put(self, element): 

        while len(self.list) >= self.max_length and self.max_length != -1: 
            time.sleep(0.1)
        
        self.lock.acquire()
        self.list.append(element)
        
        self.lock.release()
    
    def put_batch(self, batch): 

        while len(self.list) >= self.max_length and self.max_length != -1: 
            time.sleep(0.1)

        self.lock.acquire()
        self.list.extend(batch)
        self.lock.release()

    def get(self, number_of_elements): 
        
        while len(self) < number_of_elements: 
            if(self.finished):
                raise StopIteration
            time.sleep(0.1)
        
        self.lock.acquire()
        items = self.list[0:number_of_elements]
        self.list = self.list[number_of_elements:]
        self.lock.release()
        return items 


#class for providing batches of data. The number of epochs can be specified. 
class ERFH5_DataGenerator():


    #####Parameters#####
    #data_path: root directory for ERFH5 files 
    #indices: elements of filling factors sequence that should be contained in an instance
    #max_queue_length: number of instances that are preloaded to the batch queue 
    #sequence: if true, instances consist of more than one state (exact number is specified with indices) 
    def __init__(self, data_path='/home/', batch_size=64, epochs=80, indices=[0, 20, 40, 60, 80], max_queue_length=-1,
                 pipeline_mode=Pipeline_Mode.single_instance, num_validation_samples=1, num_workers=2):
        self.data_path = data_path
        self.batch_size = batch_size
        self.indices = indices
        self.pipeline_mode = pipeline_mode
        self.data_dict = dict()
        self.num_validation_samples = num_validation_samples
        self.paths = self.__get_paths_to_files(self.data_path)
        random.shuffle(self.paths)
        
        self.max_queue_length = max_queue_length
        self.path_queue = Thread_Safe_List()
        self.validation_list = []

        self.num_workers = num_workers
        self.epochs = epochs
        
        self.batch_queue = Thread_Safe_List(max_length=self.max_queue_length)
        self.threads = []
        self.barrier = threading.Barrier(self.num_workers)
        self.t_begin = 0
        self.t_end = 50
        self.t_delta = 2
        self.t_target_offset = 2
        self.t_sequnce_distance = 10
        self.t_final = 1000 

     
        try:
            self.__fill_path_queue()
        except Exception as e:
            raise e
        
        self.__fill_validation_list()

        for i in range(self.num_workers):
            t_batch = threading.Thread(target=self.__fill_batch_queue)
            t_batch.start()
            self.threads.append(t_batch)

    def get_current_queue_length(self):
        return self.batch_queue.__len__()       
    
    def __fill_validation_list(self):
        while len(self.validation_list) < self.num_validation_samples: 
            sample = self.paths[0]
            self.paths = self.paths[1:]
            if self.pipeline_mode == Pipeline_Mode.index_sequence:
                try:
                    instance = self.__create_data_instance(sample)
                except IndexError as ie:
                    continue 
                self.validation_list.append(instance)
            elif self.pipeline_mode == Pipeline_Mode.single_instance:
                try:
                    instance = self.__create_data_instance(sample)
                except IndexError as ie:
                    continue 
                instance = instance[0]
                instance = list(torch.unbind(instance))
                random.shuffle(instance)
                for i in instance:
                    self.validation_list.append(i)
            elif self.pipeline_mode == Pipeline_Mode.time_sequence:
                try:
                    instance_list = self.__create_data_instances_for_a_land_before_our_time(sample)
                except NoSequenceException as E:
                    continue
                self.validation_list.extend(instance_list)

    
    #returns a list of all file paths in a root directory including sub-directories
    def __get_paths_to_files(self, root_directory):
        dataset_filenames = []
        for (dirpath, dirnames, filenames) in walk(root_directory):
            if filenames: 
                filenames = [dirpath + '/' + f for f in filenames]
                dataset_filenames.extend(filenames)
        return dataset_filenames   

    #function that creates instances of data from filepaths 
    def __fill_batch_queue(self):
        while len(self.path_queue) > 0:
            try:
                file = self.path_queue.get(1)[0]
            except StopIteration as si:
                break


            if(len(self.path_queue) == 0):
                self.path_queue.kill()

            if self.pipeline_mode == Pipeline_Mode.index_sequence or self.pipeline_mode == Pipeline_Mode.single_instance:
                try:
                    data = self.__create_data_instance(file)
                except IndexError as ie:
                    continue 
            elif self.pipeline_mode == Pipeline_Mode.time_sequence:
                try:
                    data = self.__create_data_instances_for_a_land_before_our_time(file)
                except NoSequenceException as E:
                    continue
                self.batch_queue.put_batch(data)

            if self.pipeline_mode == Pipeline_Mode.index_sequence:
                self.batch_queue.put(data)
            elif self.pipeline_mode == Pipeline_Mode.single_instance:
                data = data[0]
                data = list(torch.unbind(data)) 
                random.shuffle(data)
                for i in data:
                    self.batch_queue.put((i,0))

        self.barrier.wait()
        self.batch_queue.kill()
        print(">>>INFO: Data loading complete. - SUCCESS")
    
    #function for providing the filepaths in a shuffeled order and for realizing epochs.
    def __fill_path_queue(self): 
        if len(self.paths) == 0:
            raise Exception("No file paths found")
        
        for i in range(self.epochs):
            new_paths = self.paths
            random.shuffle(new_paths)
            self.path_queue.put_batch(new_paths)
        
       

        
   
    #creates a instance of data and label from the preprocessed file. 

    def __create_data_instances_for_a_land_before_our_time(self, filename):
        tensor_instances = list()
        
        if filename in self.data_dict:
            if self.data_dict[filename] != -1:
                return self.data_dict[filename]
            else:
                raise NoSequenceException  
        else:  
            try:
                instanceList = self.__get_all_sequences_for_file(filename, self.t_begin, self.t_end, self.t_delta, self.t_target_offset, self.t_sequnce_distance, self.t_final)
            except NoSequenceException as E:
                self.data_dict[filename] = -1
                raise E
            
            for inst in instanceList:
                sequence, label = torch.FloatTensor(inst[0]), torch.FloatTensor(inst[1])
                tensor_instances.append((sequence,label))
            
            self.data_dict[filename] = tensor_instances
            


        return tensor_instances
        


    def __create_data_instance(self, filename):
        
        if filename in self.data_dict:
            
            data = self.data_dict[filename]
            if data == -1:
                raise IndexError
            return data
        else:
             
                try:
                    fillings, label = self.__get_states_and_fillings(filename)
                    fillings, label = torch.FloatTensor(fillings), torch.FloatTensor(label)
                    self.data_dict[filename] = (fillings,label)
                except IndexError as ie:
                    print(">>>WARNING: File", filename, "does not have enough steps.")
                    self.data_dict[filename] = -1
                    raise ie
        
        """ label = int(states_and_fillings[-1][0])
        states_and_fillings = [i[1] for i in states_and_fillings]
        
        instance = (states_and_fillings, label)
        #TODO cut out N frames 
         """
        return (fillings,label)

    def __get_all_sequences_for_file(self,filename, t_begin, t_end, t_delta, t_target_offset, t_sequence_distance, t_final):
        all_sequences = list()

        
        try:
            while t_end + t_target_offset < t_final:
                instance = self.__get_fillings_at_times(filename, t_begin,t_end,t_delta,t_end+t_target_offset)
                t_begin = t_begin +t_sequence_distance
                t_end = t_end + t_sequence_distance
                all_sequences.append(instance)
        except NoSequenceException:
            pass
        
        if len(all_sequences) == 0:
            raise NoSequenceException

        return all_sequences


    def __get_fillings_at_times(self, filename, t_start, t_finish, t_delta, t_target):
        t_now = t_start

        try:
            f = h5py.File(filename, 'r')
        except OSError as E:
            print(">>> ERROR: FILE", filename, "COULD NOT BE OPEND BY H5PY. THIS IS BAD. BIG OOooOF")
            raise NoSequenceException

        all_states = f['post']['singlestate']
        filling_factors_at_certain_times = list()
        filling_percentage = -1
    
        
        for state in all_states:
            try:
                time = f['post']['singlestate'][state]['entityresults']['NODE']['FILLING_FACTOR']['ZONE1_set1']['erfblock']['indexval'][()]
                
                if time >= t_target:
                    target_fillingstate = filling_factor =  f['post']['singlestate'][state]['entityresults']['NODE']['FILLING_FACTOR']['ZONE1_set1']['erfblock']['res'][()]
                    non_zeros = np.count_nonzero(target_fillingstate)
                    state_count = np.shape(target_fillingstate)[0]
                    filling_percentage = np.array(non_zeros/state_count)
                    t_target = 9999999
                if(time >= t_finish):
                    continue
                if(time >= t_now):
                    filling_factor =  f['post']['singlestate'][state]['entityresults']['NODE']['FILLING_FACTOR']['ZONE1_set1']['erfblock']['res'][()]
                    filling_factors_at_certain_times.append(filling_factor)   
                    t_now += t_delta
                
            except KeyError as e:
                continue
            
        
        #label = f['post']['singlestate'][j]['entityresults']['NODE']['FILLING_FACTOR']['ZONE1_set1']['erfblock']['indexval'][()]


        if(t_target != 9999999 or filling_factors_at_certain_times.__len__() != (t_finish -t_start)/t_delta):
            #print("Didn't",len(filling_factors_at_certain_times), t_target, filling_percentage)
            raise NoSequenceException


        flat_fillings = [x.flatten() for x in filling_factors_at_certain_times]
    
        #print("Worked",len(filling_factors_at_certain_times), t_target, filling_percentage)
        return flat_fillings, filling_percentage    

    #function for preprocessing the erfh5 file and extracting all states and the filling factors at this state 
    def __get_states_and_fillings(self, filename):
        
        f = h5py.File(filename, 'r')
        coord_as_np_array = f['post/constant/entityresults/NODE/COORDINATE/ZONE1_set0/erfblock/res'].value
        # Cut off last column (z), since it is filled with 1s anyway
        _coords = coord_as_np_array[:, :-1]
        all_states = f['post']['singlestate']
        j = ""
        filling_factors_at_certain_times = list()
        """ for k in all_states:
            j = k """
        
        for state in all_states:
            try:
                filling_factor =  f['post']['singlestate'][state]['entityresults']['NODE']['FILLING_FACTOR']['ZONE1_set1']['erfblock']['res'][()]
                #label = f['post']['singlestate'][state]['entityresults']['NODE']['FILLING_FACTOR']['ZONE1_set1']['erfblock']['indexval'][()]
            except KeyError as e:
                continue
            
            filling_factors_at_certain_times.append(filling_factor)   
        #label = f['post']['singlestate'][j]['entityresults']['NODE']['FILLING_FACTOR']['ZONE1_set1']['erfblock']['indexval'][()]

        flat_fillings = [x.flatten() for x in filling_factors_at_certain_times]
        last_filling = flat_fillings[-1]
        non_zeros = np.count_nonzero(last_filling)
        state_count = np.shape(last_filling)[0]
        filling_percentage = np.array(non_zeros/state_count)
        
        
        if self.pipeline_mode == Pipeline_Mode.index_sequence:
           
            try:
                flat_fillings = [flat_fillings[j] for j in self.indices]
            except IndexError as ie:
                raise ie
        
        
        return flat_fillings, filling_percentage

    def __iter__(self):
        return self 

    def __next__(self):
        try:
            batch = self.batch_queue.get(self.batch_size)
        except StopIteration as e:
            raise e 
            
        data = [i[0]for i in batch]
        labels = [i[1] for i in batch]
        data = torch.stack(data)
        labels = torch.stack(labels)
        return data, labels

    def __len__(self): 
        return self.epochs * len(self.paths)


    def get_validation_samples(self):
        return self.validation_list 
    
    ########################################################################################
    ###functions not relevant for using this file, still there in case someone needs them###
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
    data_folder = '/run/user/1001/gvfs/smb-share:server=137.250.170.56,share=share/data/RTM/Lautern/clean_erfh5/'
    #data_folder = '/home/niklas/Documents/Data'
    generator = ERFH5_DataGenerator(data_path=data_folder, batch_size=4, epochs=1, pipeline_mode = Pipeline_Mode.single_instance)
    
    batch_data, batch_labels = generator.__next__()
    
    for data, labels in generator: 
        print(data.size(), labels.size())