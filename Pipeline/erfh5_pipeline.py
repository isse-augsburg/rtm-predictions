import logging
import os
import pickle
import random
import socket
import threading
from pathlib import Path
from time import sleep

import torch

from Pipeline import data_gather as dg, data_loader_sensor as dls


class ThreadSafeList:
    """Implements a thread safe list that is much faster than built-in
    python lists.

    Args:
            max_length (int): Max length the list can have. Should be specified
            if the memory consumption is a problem.
    """

    def __init__(self):
        self.list = []
        self.lock = threading.Lock()
        self.finished = False

    # if called, the thread using this queue will commit suicide
    def kill(self):
        """Sets a flag that raises a StopIteration when get is called. Prevents
        threads from waiting infinitely
        long for enough elements.
        """
        self.finished = True

    def __len__(self):
        # self.lock.acquire()
        length = len(self.list)
        # self.lock.release()
        return length

    def randomise(self):
        """Shuffles the list every 10 seconds. Used for shuffling sequences
        that were extracted from the same file.
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
        """Appends multiple elements elementwise to the list
        (uses extend instead of append).

        Args:
            batch (array-like): List of elements that should be added
            to the list.

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
            List: list consisting of the first number_of_elements elements
            of the thread safe list.
        """

        if len(self) < number_of_elements:
            return None

        self.lock.acquire()
        items = self.list[0:number_of_elements]
        self.list = self.list[number_of_elements:]
        self.lock.release()
        return items


# def clear_last_line():
#     """Hack for deleting the last printed console line
#     """
#     sys.stdout.write("\033[F")


def assert_instance_correctness(instance):
    assert isinstance(
        instance, list
    ), '''The data loader seems to return instances in the wrong format. 
            The required format is [(data_1, label1), ... , 
            (data_n, label_n)] or None.'''
    for i in instance:
        assert (
            isinstance(i, tuple) and len(i) == 2
        ), '''The data loader seems to return instances in the wrong format. 
                The required format is [(data_1, label1), ... , 
                (data_n, label_n)] or None.'''


def transform_to_tensor_and_cache(i, num, s_path, separate_set_list):
    data, label = torch.FloatTensor(i[0]), torch.FloatTensor(i[1])
    separate_set_list.append((data, label))
    if s_path is not None:
        s_path.mkdir(parents=True, exist_ok=True)
        torch.save(data, s_path.joinpath(str(num) + "-data" + ".pt"))
        torch.save(label, s_path.joinpath(str(num) + "-label" + ".pt"))


def load_cached_data_and_label(instance_f, s_path):
    _list = []
    for i in range(len(instance_f) // 2):
        data = torch.load(s_path.joinpath(instance_f[i * 2]))
        label = torch.load(s_path.joinpath(instance_f[i * 2 + 1]))
        _list.append((data, label))
    return _list


def transform_list_of_linux_paths_to_windows(input_list):
    if socket.gethostname() == "swtse130":
        win_paths = []
        for e in input_list:
            if e[:4] == "/cfs":
                win_paths.append(Path(e.replace("/cfs/home", "X:")))
        return win_paths
    else:
        return input_list


class ERFH5DataGenerator:
    """ Iterable object that generates batches of a specified size. 

    Args: 
        data_path (string): path to the root directory of the data
        data_processing_function (function): function that transforms a file
        path to extracted data;
        MUST return the following format:
        [(data_1, label_1), ... , (data_n, label_n)]
        data_gather_function (function): function that returns a list of paths
        to all files that should be
        used for training
        batch_size (int): size of the generated batches 
        epochs (int): number of epochs 
        max_queue_length (int): restricts the number of pre-loaded batches.
        Batch_size * 4 is usually a good value
        num_validation_samples (int): number of instances that are used as
        validation samples,
        instances mean single frames, not entire runs
        num_test_samples (int): number of instances that are used as test
        samples
        num_workers (int): number of threads that transform file paths to data. 
    """

    def __init__(
            self,
            data_paths=["/home/"],
            data_processing_function=None,
            data_gather_function=None,
            batch_size=1,
            epochs=5,
            max_queue_length=8096,
            num_validation_samples=100,
            num_test_samples=100,
            num_workers=4,
            cache_path=None,
            save_path=None,
            load_datasets_path=None,
            test_mode=False,
    ):
        self.kill_t_shuffle = False
        self.kill_t_batch = False
        self.threadlist = []
        self.data_paths = [str(x) for x in data_paths]
        self.batch_size = batch_size
        self.epochs = epochs
        self.max_queue_length = max_queue_length
        assert max_queue_length > 0
        assert num_workers > 0
        self.num_validation_samples = num_validation_samples
        self.num_test_samples = num_test_samples
        self.num_workers = num_workers
        self.data_function = data_processing_function
        self.data_gather = data_gather_function
        self.cache_path = None
        self.cache_path_flist = None
        if cache_path is not None:
            self.init_cache_paths(cache_path)

        if self.data_function is None or self.data_gather is None:
            raise Exception("No data processing or reading function specified!")

        self.data_dict = dict()
        self.logger = logging.getLogger(__name__)
        self.logger.info("Gathering Data...")
        self.paths = []
        self.batch_queue = ThreadSafeList()
        self.path_queue = ThreadSafeList()
        self.validation_list = []
        self.test_list = []

        self.validation_fnames = []
        self.test_fnames = []
        self.training_fnames = []

        if not test_mode:
            self.init_generators_and_run(save_path, load_datasets_path)

    def init_cache_paths(self, cache_path):
        self.cache_path = Path(cache_path).joinpath(self.data_function.__name__)
        self.cache_path.mkdir(parents=True, exist_ok=True)
        self.cache_path_flist = Path(cache_path).joinpath("filelists")
        self.cache_path_flist.mkdir(parents=True, exist_ok=True)

    def end_threads(self):
        self.kill_t_batch = True
        self.kill_t_shuffle = True
        [x.join() for x in self.threadlist]

    def init_generators_and_run(self, save_path, load_path):
        for path in self.data_paths:
            if self.cache_path_flist is not None:
                path_name = Path(path).stem
                file = self.cache_path_flist.joinpath(path_name)
                if os.path.isfile(file):
                    with open(file, "rb") as f:
                        self.paths.extend(pickle.load(f))
                    continue
                else:
                    gatherd = self.data_gather(path)
                    with open(file, "wb") as f:
                        pickle.dump(gatherd, f)
                    self.paths.extend(gatherd)
            else:
                gatherd = self.data_gather(path)
                self.paths.extend(gatherd)
        if len(self.paths) > 1:
            random.shuffle(self.paths)
        self.logger.info("Gathering Data... Done.")
        self.barrier = threading.Barrier(self.num_workers)
        self.logger.info("Separating data sets ...")
        if load_path is None:
            self.validation_list, self.validation_fnames = \
                self.__fill_separate_set_list_from_all_paths(
                    self.num_validation_samples
                )
            self.test_list, self.test_fnames = \
                self.__fill_separate_set_list_from_all_paths(
                    self.num_test_samples
                )
        else:
            self.logger.info(f'Loading existing datasets from: {load_path}')
            self.load_data_sets(load_path)
            self.validation_list = self.__get_data_samples_from_list(
                self.validation_fnames,
                self.num_validation_samples)
            self.test_list = \
                self.__get_data_samples_from_list(self.test_fnames,
                                                  self.num_test_samples)
        if save_path is not None:
            self.save_data_sets(save_path)
        self.logger.info("Filling Path Queue...")
        try:
            self.__fill_path_queue()
        except Exception as e:
            raise e
        self.logger.info("Filling Path Queue... Done.")
        self.__print_info()
        for i in range(self.num_workers):
            t_batch = threading.Thread(target=self.__fill_batch_queue)
            self.threadlist.append(t_batch)
            t_batch.start()
        self.t_shuffle = threading.Thread(target=self.__shuffle_batch_queue)
        self.threadlist.append(self.t_shuffle)
        self.t_shuffle.start()

    def save_data_sets(self, save_path):
        with open(save_path / "validation_set.p", "wb") as f:
            pickle.dump(self.validation_fnames, f)
        with open(save_path / "test_set.p", "wb") as f:
            pickle.dump(self.test_fnames, f)
        with open(save_path / "training_set.p", "wb") as f:
            pickle.dump(self.paths, f)

    def load_data_sets(self, load_path):
        with open(load_path / "validation_set.p", 'rb') as f:
            self.validation_fnames = pickle.load(f)
        with open(load_path / "test_set.p", 'rb') as f:
            self.test_fnames = pickle.load(f)
        with open(load_path / "training_set.p", 'rb') as f:
            self.paths = pickle.load(f)
        if socket.gethostname() == 'swtse130':
            self.validation_fnames = transform_list_of_linux_paths_to_windows(
                self.validation_fnames)
            self.test_fnames = transform_list_of_linux_paths_to_windows(
                self.test_fnames)
            self.paths = transform_list_of_linux_paths_to_windows(self.paths)

    def __shuffle_batch_queue(self):
        while (
                not self.kill_t_shuffle
                and (len(self.path_queue) > self.batch_size
                     or len(self.batch_queue) > self.batch_size)
        ):
            self.batch_queue.randomise()
            sleep(10)

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
        self.logger.info("###########################################")
        self.logger.info(">>> Generator INFO <<<")
        self.logger.info(f"Used data folders:")
        for e in self.data_paths:
            self.logger.info(e)
        self.logger.info(f"Used data gather function: {self.data_gather}")
        self.logger.info(f"Used data processing function: {self.data_function}")
        self.logger.info(f"Number of epochs: {self.epochs}")
        self.logger.info(f"Batchsize: {self.batch_size}")
        self.logger.info(f"Number of unique samples: {len(self.paths)}")
        self.logger.info(f"Number of total samples: {self.__len__()}")
        self.logger.info(
            f"Number of validation samples: {self.num_validation_samples}")
        self.logger.info("###########################################")

    def __get_data_samples_from_list(self, input_list, wanted_len):
        separate_set_list = []

        for sample in input_list:
            instance = self.data_function(sample)

            if instance is None:
                continue
            else:
                assert_instance_correctness(instance)
                for num, i in enumerate(instance):
                    transform_to_tensor_and_cache(i, num, None,
                                                  separate_set_list)
                    if len(separate_set_list) == wanted_len:
                        break

        return separate_set_list

    def __fill_separate_set_list_from_all_paths(self, wanted_len):
        separate_set_list = []
        separate_fname_list = []
        if len(self.paths) == 0:
            raise Exception("No file paths found")

        while len(separate_set_list) < wanted_len:
            s_path = None
            # If IndexError here: files are all too short
            sample = self.paths[0]
            separate_fname_list.append(sample)
            self.paths = self.paths[1:]
            if self.cache_path is not None:
                s_path = Path(sample)
                s_path = self.cache_path.joinpath(s_path.stem)
                if s_path.exists():
                    instance_f = s_path.glob("*.pt")
                    instance_f = sorted(instance_f)
                    separate_set_list.extend(
                        load_cached_data_and_label(instance_f, s_path))
                    continue
                else:
                    s_path.mkdir(parents=True, exist_ok=True)

            instance = self.data_function(sample)

            # data_function must return [(data, label) ... (data, label)]
            if instance is None:
                continue
            else:
                assert_instance_correctness(instance)
                for num, i in enumerate(instance):
                    transform_to_tensor_and_cache(i, num, s_path,
                                                  separate_set_list)
                    if len(separate_set_list) == wanted_len:
                        break

        return separate_set_list, separate_fname_list

    def __fill_batch_queue(self):
        while not self.kill_t_batch and (
                len(self.batch_queue) < self.max_queue_length):
            s_path = None
            if len(self.path_queue) < self.batch_size:
                return

            file = self.path_queue.get(1)
            if file is None:
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
                    if s_path.exists():
                        instance_f = s_path.glob("*.pt")
                        instance_f = sorted(instance_f)

                        instance = load_cached_data_and_label(instance_f,
                                                              s_path)
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
                    assert_instance_correctness(instance)
                    tensor_instances = list()

                    for num, i in enumerate(instance):
                        transform_to_tensor_and_cache(i, num, s_path,
                                                      tensor_instances)
                    self.batch_queue.put_batch(tensor_instances)
                    self.data_dict[file] = tensor_instances

    def __iter__(self):
        return self

    def __next__(self):
        if (
                len(self.path_queue) < self.batch_size
                and len(self.batch_queue) < self.batch_size
        ):
            raise StopIteration

        while len(self.batch_queue) < self.batch_size:
            if len(self.path_queue) < self.batch_size:
                raise StopIteration
            sleep(0.1)
        batch = self.batch_queue.get(self.batch_size)
        if len(self.batch_queue) < self.max_queue_length / 4:
            if (
                    threading.active_count() < self.num_workers + 1
                    and len(self.path_queue) > self.batch_size
            ):

                for _ in range(self.num_workers):
                    t_batch = threading.Thread(target=self.__fill_batch_queue)
                    t_batch.start()

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
            List: list containing self.num_validation_samples
            instances for validation.
        """
        return self.validation_list

    def get_test_samples(self):
        """
        Returns:
            List: list containing self.num_test_samples
            instances for validation.
        """
        return self.test_list

    def load_test_set(self, path):
        self.test_list = pickle.load(open(path, "rb"))


if __name__ == "__main__":
    # generator = ERFH5_DataGenerator(data_path=
    # ["/cfs/home/s/c/schroeni/Git/tu-kaiserslautern-data/Images"],
    # batch_size=1, epochs=2, max_queue_length=16,
    # data_processing_function=get_image_state_sequence,
    # data_gather_function=get_folders_within_folder) """
    # '/run/user/1001/gvfs/smb-share:server=137.250.170.56,
    # share=share/data/RTM/Lautern/1_solved_simulations/20_auto_solver_inputs/'
    # '/run/user/1001/gvfs/smb-share:server=137.250.170.56,
    # share=share/data/RTM/Lautern/clean_erfh5/'
    generator = ERFH5DataGenerator(
        data_paths=[
            "/run/user/1001/gvfs/smb-share:server=137.250.170.56,"
            "share=share/data/RTM/Lautern/output/with_shapes/"
            "2019-05-13_16-28-01_200p/0"
        ],
        data_processing_function=dls.get_sensordata_and_filling_percentage,
        data_gather_function=dg.get_filelist_within_folder,
        batch_size=1,
        epochs=2,
        max_queue_length=16,
    )
    for data, label in generator:
        print(data.size(), label.size())
