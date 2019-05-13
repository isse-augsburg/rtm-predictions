from os import listdir, walk
import data_loaders as dl
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
        # self.lock.acquire()
        length = len(self.list)
        # self.lock.release()
        return length

    def randomise(self):
        while not self.finished:
            self.lock.acquire()
            random.shuffle(self.list)
            print(">>>INFO: Successfully shuffeled batch queue")
            self.lock.release()
            time.sleep(10)

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

    def __init__(self, data_path=['/home/'], data_processing_function=None, data_gather_function=None, batch_size=64,  epochs=80, max_queue_length=-1, num_validation_samples=1, num_workers=4):
        self.data_path = data_path
        self.batch_size = batch_size
        self.epochs = epochs
        self.max_queue_length = max_queue_length
        self.num_validation_samples = num_validation_samples
        self.num_workers = num_workers
        self.data_function = data_processing_function
        self.data_gather = data_gather_function

        if self.data_function is None or self.data_gather is None:
            raise Exception(
                "No data processing or reading function specified!")

        self.data_dict = dict()
        print(">>> Generator: Gathering Data...")
        self.paths = self.data_gather(self.data_path)
        random.shuffle(self.paths)

        
        self.batch_queue = Thread_Safe_List(max_length=self.max_queue_length)
        self.path_queue = Thread_Safe_List()
        self.validation_list = []
        self.barrier = threading.Barrier(self.num_workers)
        print(">>> Generator: Filling Validation List...")
        self.__fill_validation_list()

        print(">>> Generator: Filling Path Queue...")
        try:
            self.__fill_path_queue()
        except Exception as e:
            raise e
    
        for i in range(self.num_workers):
            t_batch = threading.Thread(target=self.__fill_batch_queue)
            t_batch.start()
        self.t_shuffle = threading.Thread(target=self.__shuffle_batch_queue)
        self.t_shuffle.start()
   

    def __shuffle_batch_queue(self):
        
        self.batch_queue.randomise()


    def __fill_path_queue(self):
        if len(self.paths) == 0:
            raise Exception("No file paths found")

        for i in range(self.epochs):
            new_paths = self.paths
            random.shuffle(new_paths)
            self.path_queue.put_batch(new_paths)
    
    def get_current_queue_length(self):
        return self.batch_queue.__len__()   

    def __fill_validation_list(self):
        if len(self.paths) == 0:
            raise Exception("No file paths found")

        while len(self.validation_list) < self.num_validation_samples:

            #If IndexError here: files are all too short 
            sample = self.paths[0]			
            self.paths = self.paths[1:]
            instance = self.data_function(sample)

            # data_function must return [(data, label) ... (data, label)]
            if instance is None:
                continue
            else:
                for i in instance:
                    data, label = torch.FloatTensor(
                            i[0]), torch.FloatTensor(i[1])
                    self.validation_list.append((data, label))

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
                # data_function must return [(data, label) ... (data, label)]
                instance = self.data_function(file)

                if instance is None:
                    self.data_dict[file] = None
                    continue
                else:
                    tensor_instances = list()

                    for i in instance:
                        data, label = torch.FloatTensor(
                            i[0]), torch.FloatTensor(i[1])
                        self.batch_queue.put((data, label))
                        tensor_instances.append((data, label))

                    self.data_dict[file] = tensor_instances

        # TODO
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

    """ generator = ERFH5_DataGenerator(data_path= ["/cfs/home/s/c/schroeni/Git/tu-kaiserslautern-data/Images"],
                                    batch_size=1, epochs=2, max_queue_length=16, data_processing_function=get_image_state_sequence, data_gather_function=get_folders_within_folder) """
    #'/run/user/1001/gvfs/smb-share:server=137.250.170.56,share=share/data/RTM/Lautern/1_solved_simulations/20_auto_solver_inputs/'
    #'/run/user/1001/gvfs/smb-share:server=137.250.170.56,share=share/data/RTM/Lautern/clean_erfh5/'
    print("sers")
    generator = ERFH5_DataGenerator(data_path = ['/run/user/1001/gvfs/smb-share:server=137.250.170.56,share=share/data/RTM/Lautern/1_solved_simulations/20_auto_solver_inputs/'],
        data_processing_function=dl.get_sensordata_and_filling_percentage, data_gather_function=dl.get_filelist_within_folder, batch_size=2, epochs=2, max_queue_length=16)
    for data, label in generator:
        print(data.size(), label.size())
