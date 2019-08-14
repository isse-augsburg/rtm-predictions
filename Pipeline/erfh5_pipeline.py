from Pipeline import data_loaders as dl, data_gather as dg, data_loader_sensor as dls
#import data_loaders as dl, data_gather as dg, data_loader_sensor as dls
import threading
import time
import pickle
import random
import torch
import sys
import os
from pathlib import Path

class Thread_Safe_List():
    """Implements a thread safe list that is much faster than built-in python lists. 

    Args:
            max_length (int): Max length the list can have. Should be specified if the memory consumption is a problem.
    """

    def __init__(self):
        
        self.list = []
        self.lock = threading.Lock()
        self.finished = False

    # if called, the thread using this queue will commit suicide
    def kill(self):
        """Sets a flag that raises a StopIteration when get is called. Prevents threads from waiting infitinely long for enough elements. 
        """
        self.finished = True

    def __len__(self):
        # self.lock.acquire()
        length = len(self.list)
        # self.lock.release()
        return length

    def randomise(self):
        """Shuffles the list every 10 seconds. Used for shuffling sequences that were extracted from the same file.
        """
    
        self.lock.acquire()
        random.shuffle(self.list)
        # print(">>>INFO: Successfully shuffeled batch queue")
        self.lock.release()
            

    def put(self, element):
        """Appends a single element to the list. 
        
        Args:
            element: Element that should be added.
        """
       

        self.lock.acquire()
        self.list.append(element)

        self.lock.release()

    def put_batch(self, batch):
        """Appends multiple elements elementwise to the list (uses extend instead of append). 

        Args:
            batch (array-like): List of elements that should be added to the list. 

        Example:
            >>> list = [1, 2, 3, 4]
            >>> list.put_batch([5, 6]) 
            >>> print(list) 
            [1, 2, 3, 4, 5, 6]
            >>> list.put([7, 8])
            >>> print(list) 
            [1, 2, 3, 4, 5, 6, [7, 8]]
        """
       
        self.lock.acquire()
        self.list.extend(batch)
        self.lock.release()


    def get(self, number_of_elements):
        """
        Args:
            number_of_elements (int): number of elements that should be returned
        
        Returns:
            List: list consisting of the first number_of_elements elements of the thread safe list.

        
        """

        if len(self) < number_of_elements:
           return None

        self.lock.acquire()
        items = self.list[0:number_of_elements]
        self.list = self.list[number_of_elements:]
        self.lock.release()
        return items

   

def clear_last_line(): 
    """Hack for deleting the last printed console line
    """
    sys.stdout.write("\033[F") 


class ERFH5_DataGenerator():
    """ Iterable object that generates batches of a specified size. 

    Args: 
        data_path (string): path to the root directory of the data
        data_processing_function (function): function that transforms a file path to extracted data; MUST return the following format: [(data_1, label_1), ... , (data_n, label_n)]
        data_gather_function (function): function that returns a list of paths to all files that should be used for training 
        batch_size (int): size of the generated batches 
        epochs (int): number of epochs 
        max_queue_length (int): restricts the number of pre-loaded batches. Batch_size * 4 is usually a good value
        num_validation_samples (int): number of instances that are used as validation samples. 
        num_workers (int): number of threads that transform file paths to data. 
    """

    def __init__(self, data_paths=['/home/'], data_processing_function=None, data_gather_function=None, batch_size=64,
                 epochs=80, max_queue_length=512, num_validation_samples=1, num_workers=4, cache_path=None):
        self.data_paths = [str(x) for x in data_paths]
        self.batch_size = batch_size
        self.epochs = epochs
        self.max_queue_length = max_queue_length
        assert(max_queue_length>0)
        self.num_validation_samples = num_validation_samples
        self.num_workers = num_workers
        self.data_function = data_processing_function
        self.data_gather = data_gather_function
        self.cache_path=None
        self.cache_path_flist = None
        if cache_path is not None:
            self.cache_path = Path(cache_path).joinpath(self.data_function.__name__)
            self.cache_path.mkdir(parents=True, exist_ok=True)
            self.cache_path_flist = Path(cache_path).joinpath("filelists")
            self.cache_path_flist.mkdir(parents=True, exist_ok=True)

        if self.data_function is None or self.data_gather is None:
            raise Exception(
                "No data processing or reading function specified!")

        self.data_dict = dict()
        print(">>> Generator: Gathering Data...")
        self.paths = []
        for path in self.data_paths:
            if self.cache_path_flist is not None:
                path_name = path.split("/")[-1]
                file = self.cache_path_flist.joinpath(path_name)
                if(os.path.isfile(file)):
                    self.paths.extend(pickle.load(open(file, "rb")))
                    continue
                else:
                    gatherd = self.data_gather(path)
                    pickle.dump(gatherd,open(file,"wb"))
                    self.paths.extend(gatherd)
            else:
                gatherd = self.data_gather(path)
                self.paths.extend(gatherd)
            
        

        self.paths = self.paths
        if len(self.paths) > 1:
            random.shuffle(self.paths)
        clear_last_line()
        print(">>> Generator: Gathering Data... Done.")
        self.batch_queue = Thread_Safe_List()
        self.path_queue = Thread_Safe_List()
        self.validation_list = []
        self.test_list = []
        self.barrier = threading.Barrier(self.num_workers)
        
        print(">>> Generator: Filling Validation List...")
        self.__fill_validation_list()
        clear_last_line()
        print(">>> Generator: Filling Validation List... Done.")

        print(">>> Generator: Filling Path Queue...")
        try:
            self.__fill_path_queue()
        except Exception as e:
            raise e
        clear_last_line()
        
        print(">>> Generator: Filling Path Queue... Done.")
        
        self.__print_info()
        
        for i in range(self.num_workers):
            t_batch = threading.Thread(target=self.__fill_batch_queue)
            t_batch.start()
        self.t_shuffle = threading.Thread(target=self.__shuffle_batch_queue)
        self.t_shuffle.start()

    def __shuffle_batch_queue(self):
        while(len(self.path_queue) > self.batch_size or len(self.batch_queue) > self.batch_size):
            self.batch_queue.randomise()
            time.sleep(10)

    def __fill_path_queue(self):
        if len(self.paths) == 0:
            raise Exception("No file paths found")

        for i in range(self.epochs):
            new_paths = self.paths
            random.shuffle(new_paths)
            self.path_queue.put_batch(new_paths)

    def get_current_queue_length(self):
        """
        Returns: 
            Int: current number of pre-loaded batches
        """

        return self.batch_queue.__len__()

    def __print_info(self):
        print("###########################################")
        print(">>> Generator INFO <<<")
        print("Used data folders:", self.data_paths)
        print("Used data gather function:", self.data_gather)
        print("Used data processing function:", self.data_function)
        print("Number of epochs:", self.epochs)
        print("Batchsize:", self.batch_size)
        print("Number of unique samples:", len(self.paths))
        print("Number of total samples:", self.__len__())
        print("Number of validation samples:", self.num_validation_samples)
        print("###########################################")



    def __fill_validation_list(self):
        if len(self.paths) == 0:
            raise Exception("No file paths found")

        while len(self.validation_list) < self.num_validation_samples:
            s_path = None
            # If IndexError here: files are all too short
            sample = self.paths[0]
            self.paths = self.paths[1:]
            if self.cache_path is not None:
                s_path = Path(sample)
                s_path = self.cache_path.joinpath(s_path.stem)
                if(s_path.exists()):
                    instance_f = s_path.glob("*.pt")
                    instance_f = sorted(instance_f)
                    for i in range(len(instance_f)//2):
                        data = torch.load(s_path.joinpath(instance_f[i*2]))
                        label = torch.load(s_path.joinpath(instance_f[i*2+1]))
                        self.validation_list.append((data, label))
                    continue
                else:
                    s_path.mkdir(parents=True, exist_ok=True)
                
                   
           
            instance = self.data_function(sample)
            
            # data_function must return [(data, label) ... (data, label)]
            if instance is None:
                continue
            else:
                assert isinstance(instance, list), "The data loader seems to return instances in the wrong format. The required format is [(data_1, label1), ... , (data_n, label_n)] or None."
                for i in instance:
                    assert isinstance(i, tuple) and len(i) == 2,"The data loader seems to return instances in the wrong format. The required format is [(data_1, label1), ... , (data_n, label_n)] or None."
        
                for num, i in enumerate(instance):
                    data, label = torch.FloatTensor(
                        i[0]), torch.FloatTensor(i[1])
                    self.validation_list.append((data, label))
                    if s_path is not None:
                        torch.save(data, s_path.joinpath(str(num)+"-data"+ ".pt"))
                        torch.save(label, s_path.joinpath(str(num)+"-label"+ ".pt"))



    def __fill_batch_queue(self):
        
        while len(self.batch_queue) < self.max_queue_length:
            s_path = None
            if(len(self.path_queue) < self.batch_size):
               # print(">>>INFO: Thread ended - At Start")
                return
           
            file = self.path_queue.get(1)
            if file is None:
               # print(">>>INFO: Thread ended - At File")
                return 
            file = file[0]
            

            if file in self.data_dict:
                instance = self.data_dict[file]

                if instance is None:
                    continue
                self.batch_queue.put_batch(instance)

            else:
                # data_function must return [(data, label) ... (data, label)]
                if self.cache_path is not None:
                    s_path = Path(file)
                    s_path = self.cache_path.joinpath(s_path.stem)
                    if(s_path.exists()):
                        instance_f = s_path.glob("*.pt")
                        instance_f = sorted(instance_f)
                        instance = []
                        for i in range(len(instance_f)//2):
                            data = torch.load(s_path.joinpath(instance_f[i*2]))
                            label = torch.load(s_path.joinpath(instance_f[i*2+1]))
                            instance.append((data, label))
                        self.batch_queue.put_batch(instance)
                        self.data_dict[file] = instance
                        continue

                    else:
                        s_path.mkdir(parents=True, exist_ok=True)

                instance = self.data_function(file)
                

                if instance is None:
                    self.data_dict[file] = None
                    continue
                else:
                    assert isinstance(instance, list), "The data loader seems to return instances in the wrong format. The required format is [(data_1, label1), ... , (data_n, label_n)]."
                    for i in instance:
                        assert isinstance(i, tuple) and len(i) == 2,"The data loader seems to return instances in the wrong format. The required format is [(data_1, label1), ... , (data_n, label_n)]."
        
                    tensor_instances = list()

                    for num, i in enumerate(instance):
                        data, label = torch.FloatTensor(i[0]), torch.FloatTensor(i[1])
                       
                        tensor_instances.append((data, label))
                        if s_path is not None:
                            torch.save(data, s_path.joinpath(str(num)+"-data"+ ".pt"))
                            torch.save(label, s_path.joinpath(str(num)+"-label"+ ".pt"))
                    self.batch_queue.put_batch(tensor_instances)
                    self.data_dict[file] = tensor_instances

        
        print(">>>INFO: Thread ended - At End")




    def __iter__(self):
        return self

    def __next__(self):
        if(len(self.path_queue) < self.batch_size and len(self.batch_queue) < self.batch_size):
            raise StopIteration
        
        while(len(self.batch_queue) < self.batch_size):
            if(len(self.path_queue) < self.batch_size):
                raise StopIteration
            time.sleep(0.1)
        batch = self.batch_queue.get(self.batch_size)
        if len(self.batch_queue) < self.max_queue_length/4:
            if threading.active_count() < self.num_workers + 1 and len(self.path_queue) > self.batch_size:
                #print("Starting new Threads", threading.active_count())
            
                for _ in range(self.num_workers):
                    t_batch = threading.Thread(target=self.__fill_batch_queue)
                    t_batch.start()
            

        data = [i[0] for i in batch]
        labels = [i[1] for i in batch]
        
        # FIXME does not work for batchsize > 1 if sizes of data are different - NS: This is not a bug, this is intended.
        data = torch.stack(data)
        labels = torch.stack(labels)
        return data, labels

   

    def __len__(self):
        return self.epochs * len(self.paths)

    def get_validation_samples(self):
        """
        Returns: 
            List: list containing self.num_validation_samples instances for validation. 
        """
        return self.validation_list


if __name__ == "__main__":

    #generator = ERFH5_DataGenerator(data_path= ["/cfs/home/s/c/schroeni/Git/tu-kaiserslautern-data/Images"],
                                    #batch_size=1, epochs=2, max_queue_length=16, data_processing_function=get_image_state_sequence, data_gather_function=get_folders_within_folder) """
    # '/run/user/1001/gvfs/smb-share:server=137.250.170.56,share=share/data/RTM/Lautern/1_solved_simulations/20_auto_solver_inputs/'
    # '/run/user/1001/gvfs/smb-share:server=137.250.170.56,share=share/data/RTM/Lautern/clean_erfh5/'
    generator = ERFH5_DataGenerator(data_paths=[
        '/run/user/1001/gvfs/smb-share:server=137.250.170.56,share=share/data/RTM/Lautern/output/with_shapes/2019-05-13_16-28-01_200p/0'],
                                    data_processing_function=dls.get_sensordata_and_filling_percentage,
                                    data_gather_function=dg.get_filelist_within_folder, batch_size=1, epochs=2,
                                    max_queue_length=16)
    for data, label in generator:
        print(data.size(), label.size())
        