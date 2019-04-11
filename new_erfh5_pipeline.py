from os import listdir, walk 
import threading
import time 
import random 
from enum import Enum
import torch 

import data_loaders as loaders

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


class ERFH5_DataGenerator():

    def __init__(self, data_path='/home/', data_function=None, batch_size=64,  epochs=80, max_queue_length=-1, num_validation_samples=1, num_workers=2):
        self.data_path = data_path
        self.batch_size = batch_size
        self.epochs = epochs
        self.max_queue_length = max_queue_length
        self.num_validation_samples = num_validation_samples
        self.num_workers = num_workers
        self.data_function = data_function

        if self.data_function == None:
            raise Exception("No data_function specified!")
        

        
        self.data_dict = dict()
        self.paths = self.__get_paths_to_files(self.data_path) 

        self.batch_queue = Thread_Safe_List(max_length=self.max_queue_length)
        self.path_queue = Thread_Safe_List()
        self.validation_list = []
        self.barrier = threading.Barrier(self.num_workers)

        try:
            self.__fill_path_queue()
        except Exception as e:
            raise e

        self.__fill_validation_list()

        for i in range(self.num_workers):
            t_batch = threading.Thread(target=self.__fill_batch_queue)
            t_batch.start()


    def __get_paths_to_files(self, root_directory):
        dataset_filenames = []
        for (dirpath, dirnames, filenames) in walk(root_directory):
            if filenames: 
                filenames = [dirpath + '/' + f for f in filenames]
                dataset_filenames.extend(filenames)
        return dataset_filenames  

    def __fill_path_queue(self): 
        if len(self.paths) == 0:
            raise Exception("No file paths found")
        
        for i in range(self.epochs):
            new_paths = self.paths
            random.shuffle(new_paths)
            self.path_queue.put_batch(new_paths)

    def __fill_validation_list(self): 
        while len(self.validation_list) < self.num_validation_samples: 
            sample = self.paths[0]
            self.paths = self.paths[1:]
            instance = self.data_function(sample)
            
            #data_function must return [(data, label) ... (data, label)]
            if instance is None: 
                continue
            else:
                for i in instance:
                    self.validation_list.append(i)
        

    def __fill_batch_queue(self): 
        

        while len(self.path_queue) > 0:
            try:
                file = self.path_queue.get(1)[0]
            except StopIteration as si:
                break
            
            if(len(self.path_queue) == 0):
                self.path_queue.kill()

            if file in self.data_dict:
                instance = self.data_dict[file]

                if instance is None:
                    continue
                self.batch_queue.put_batch(instance)

            else:
                #data_function must return [(data, label) ... (data, label)]
                instance = self.data_function(file)

                if instance is None:
                    self.data_dict[file] = None
                    continue
                else:
                    tensor_instances = list()

                    for i in instance:
                        data, label = torch.FloatTensor(i[0]), torch.FloatTensor(i[1])
                        self.batch_queue.put((data, label))
                        tensor_instances.append((data,label))
                    
                    self.data_dict[file] = tensor_instances
 

        #TODO 
        self.barrier.wait()
        self.batch_queue.kill()
        print(">>>INFO: Data loading complete. - SUCCESS")

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
        
if __name__ == "__main__":
    ff_loader = loaders.ERFH5_Filling_Factors_Loader()

    
    generator = ERFH5_DataGenerator(data_path='/run/user/1001/gvfs/smb-share:server=137.250.170.56,share=share/data/RTM/Lautern/clean_erfh5/', 
    batch_size=1, epochs=2, max_queue_length=16, data_function=ff_loader.get_all_sequences_for_file)

    for data, label in generator:
        print(data.size(), label.size())
    