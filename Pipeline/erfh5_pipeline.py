from Pipeline import data_loaders as dl, data_gather as dg, data_loader_sensor as dls
#import data_loaders as dl, data_gather as dg, data_loader_sensor as dls
import threading
import time
import random
import torch
import sys


class Thread_Safe_List():
    """Implements a thread safe list that is much faster than built-in python lists. 

    Args:
            max_length (int): Max length the list can have. Should be specified if the memory consumption is a problem.
    """

    def __init__(self, max_length=-1):
        
        self.list = []
        self.lock = threading.Lock()
        self.max_length = max_length
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
        while not self.finished:
            self.lock.acquire()
            random.shuffle(self.list)
            print(">>>INFO: Successfully shuffeled batch queue")
            self.lock.release()
            time.sleep(10)

    def put(self, element):
        """Appends a single element to the list. 
        
        Args:
            element: Element that should be added.
        """
        while len(self.list) >= self.max_length and self.max_length != -1:
            time.sleep(0.1)

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
        while len(self.list) >= self.max_length and self.max_length != -1:
            time.sleep(0.1)

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

        while len(self) < number_of_elements:
            if (self.finished):
                raise StopIteration
            time.sleep(0.1)

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

    def __init__(self, data_path=['/home/'], data_processing_function=None, data_gather_function=None, batch_size=64,
                 epochs=80, max_queue_length=-1, num_validation_samples=1, num_workers=4):
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
        clear_last_line()
        print(">>> Generator: Gathering Data... Done.")
        self.batch_queue = Thread_Safe_List(max_length=self.max_queue_length)
        self.path_queue = Thread_Safe_List()
        self.validation_list = []
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

        self.batch_queue.randomise()

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
        print("Used data folders:", self.data_path)
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

            # If IndexError here: files are all too short
            sample = self.paths[0]
            self.paths = self.paths[1:]
            instance = self.data_function(sample)
            
            # data_function must return [(data, label) ... (data, label)]
            if instance is None:
                continue
            else:
                assert isinstance(instance, list), "The data loader seems to return instances in the wrong format. The required format is [(data_1, label1), ... , (data_n, label_n)] or None."
                for i in instance:
                    assert isinstance(i, tuple) and len(i) == 2,"The data loader seems to return instances in the wrong format. The required format is [(data_1, label1), ... , (data_n, label_n)] or None."
        
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

            if (len(self.path_queue) == 0):
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
                    assert isinstance(instance, list), "The data loader seems to return instances in the wrong format. The required format is [(data_1, label1), ... , (data_n, label_n)]."
                    for i in instance:
                        assert isinstance(i, tuple) and len(i) == 2,"The data loader seems to return instances in the wrong format. The required format is [(data_1, label1), ... , (data_n, label_n)]."
        
                    tensor_instances = list()

                    for i in instance:
                        data, label = torch.FloatTensor(
                            i[0]), torch.FloatTensor(i[1])
                        self.batch_queue.put((data, label))
                        tensor_instances.append((data, label))

                    self.data_dict[file] = tensor_instances

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

        data = [i[0] for i in batch]
        labels = [i[1] for i in batch]
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
    generator = ERFH5_DataGenerator(data_path=[
        '/run/user/1001/gvfs/smb-share:server=137.250.170.56,share=share/data/RTM/Lautern/1_solved_simulations/20_auto_solver_inputs/'],
                                    data_processing_function=dls.get_sensordata_and_filling_percentage,
                                    data_gather_function=dg.get_filelist_within_folder, batch_size=1, epochs=2,
                                    max_queue_length=16)
    for data, label in generator:
        print(data.size(), label.size())
        