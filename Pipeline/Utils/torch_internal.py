import logging
import math
import os
import pickle
from enum import Enum
from pathlib import Path
from queue import Queue
import random

import numpy as np
import torch


class FileSetIterator:
    """ An iterator for samples stored in a set of files.
    The FileSetIterator provides an iterator over the samples stored in a set of files.
    These files are usually HDF5-files.

    Args:
        files (list of str): A list of paths to the files to be loaded
        load_data (function): A function that can load a list of samples given a filename 
            MUST return the following format:
            [(data_1, label_1), ... , (data_n, label_n)]
        cache_path (Path): A path to cache loaded samples
        worker_id (int): The id of this worker for multiprocessing environments
    """
    def __init__(self, files, load_data, cache_path=None, worker_id=0):
        self.files = files
        self.load_data = load_data
        self.cache_path = cache_path
        self.sample_queue = Queue()
        self.worker_id = worker_id

    def _assert_instance_correctness(self, instance):
        assert isinstance(
            instance, list
        ), '''The data loader seems to return instances in the wrong format. 
                The required format is [(data_1, label1), ... , 
                (data_n, label_n)] or None.'''
        for i in instance:
            assert (isinstance(i, tuple) and len(i) == 2), \
                '''The data loader seems to return instances in the wrong format. 
                    The required format is [(data_1, label1), ... , 
                    (data_n, label_n)] or None.'''

    def _get_cache_path_for_file(self, filename):
        s_path = Path(filename)
        s_path = self.cache_path.joinpath(s_path.stem)
        return s_path

    def _load_cached_samples(self, filename):
        if self.cache_path is not None:
            s_path = self._get_cache_path_for_file(filename)
            if s_path.exists():
                # Get all pickled sample files
                instance_f = s_path.glob("*.pt")
                instance_f = sorted(instance_f)
                for i in range(len(instance_f) // 2):
                    _data = torch.load(s_path.joinpath(instance_f[i * 2]))
                    _label = torch.load(s_path.joinpath(instance_f[i * 2 + 1]))
                    self.sample_queue.put((_data, _label))
                return True
        return False

    def _transform_to_tensor_and_cache(self, i, num, s_path):
        _data = torch.FloatTensor(i[0])
        # The following if else is necessary to have 0, 1 Binary Labels in Tensors
        # since FloatTensor(0) = FloatTensor([])
        if type(i[1]) is np.ndarray and len(i[1]) > 1:
            _label = torch.FloatTensor(i[1])
        else:
            if i[1] == 0:
                _label = torch.FloatTensor([0.])
            elif i[1] == 1:
                _label = torch.FloatTensor([1.])

        self.sample_queue.put((_data, _label))
        if s_path is not None:
            torch.save(_data, s_path.joinpath(f"{num}-data.pt"))
            torch.save(_label, s_path.joinpath(f"{num}-label.pt"))

    def _load_file(self):
        while True:
            if len(self.files) == 0:
                return False
            fn = self.files.pop(0)
            if self._load_cached_samples(fn):
                break  # This file was already cached; nothing to do here

            instance = self.load_data(fn)
            if instance is None:
                continue
            else:
                self._assert_instance_correctness(instance)
                s_path = None
                if self.cache_path is not None:
                    s_path = self._get_cache_path_for_file(fn)
                    s_path.mkdir(parents=True, exist_ok=True)
                for num, i in enumerate(instance):
                    self._transform_to_tensor_and_cache(i, num, s_path)
                break
        return True

    def get_remaining_files(self):
        """ Get the list of remaining files

        Returns:
            A list of remaining files.
        """
        return self.files

    def __next__(self):
        """ Get the next sample.
        This will either return a sample from the internal queue or load the next file
        from the fileset.
        When the queue is exhausted and no more files are available, it will raise a
        StopIteration.

        Raises:
            StopIteration: If no more samples are available
        """
        if self.sample_queue.empty():
            if not self._load_file():
                raise StopIteration
        return self.sample_queue.get()


class CachingMode(Enum):
    Nothing = 1
    Both = 2
    FileList = 3


class FileSetIterable(torch.utils.data.IterableDataset):
    """ An Iterable meant to be used with the torch DataLoader.

    Args:
        files: A list of (typically HDF5 files) to load
        load_data (function): A function that can load a list of samples given a filename 
            MUST return the following format:
            [(data_1, label_1), ... , (data_n, label_n)]
        cache_path (Path): A path to cache loaded samples
        cache_mode (CachingMOde): A path to cache loaded samples
    """
    def __init__(self, files, load_data, cache_path=None, cache_mode=CachingMode.Both):
        self.cache_path = cache_path
        self.load_data = load_data
        self.files = files

        self.sample_cache_path = None
        if cache_path is not None and cache_mode in [CachingMode.Both]:
            self.sample_cache_path = Path(cache_path).joinpath(self.load_data.__name__)
            self.sample_cache_path.mkdir(parents=True, exist_ok=True)

    def __iter__(self):
        """ Creates an iterator that loads a subset of the file set.
        If torch indicates a multi-worker scenario, we split the files evenly along workers.
        If some files contain significantly less samples than other files, this will lead
        to an uneven split of workload.

        If torch is not using multiprocessing, a single single Iterator will be used to
        load all files.

        Returns:
            A FileSetIterator for a subset of files.
        """
        worker_info = torch.utils.data.get_worker_info()
        worker_id = 0
        if worker_info is None:  # single-process data loading, return the full iterator
            worker_paths = self.files
        else:  # in a worker process
            # split workload
            per_worker = int(math.ceil(len(self.files) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            if worker_id == 0:
                logger = logging.getLogger(__name__)
                logger.debug(f"Each worker will process up to {per_worker} files.")
            iter_start = worker_id * per_worker
            iter_end = iter_start + per_worker
            worker_paths = self.files[iter_start:iter_end]
        return FileSetIterator(worker_paths, self.load_data, cache_path=self.sample_cache_path, worker_id=worker_id)


class FileDiscovery:
    """ A helper class to gather files from a set of base paths
    This class can be used to discover sample files in a set of directories.

    Args:
        gather_data (function): A callable that gathers files given a single root directory.
            data_gather.get_filelist_within_folder is usually used for this.
        cache_path (str): A directory to use for caching file lists, if required.
    """
    def __init__(self, gather_data, cache_path=None, cache_mode=CachingMode.FileList):
        self.filelist_cache_path = None
        if cache_path is not None and cache_mode in [CachingMode.Both, CachingMode.FileList]:
            self.filelist_cache_path = Path(cache_path).joinpath("filelists")
            self.filelist_cache_path.mkdir(parents=True, exist_ok=True)
        self.gather_data = gather_data

    def discover(self, data_paths):
        """ Get a list of files for the given set of paths.

        Args:
            data_paths (list of str): The set of paths to load

        Returns:
            A list of files that were found
        """
        paths = []
        for path in data_paths:
            if self.filelist_cache_path is not None:
                path_name = Path(path).stem
                cachefile = self.filelist_cache_path.joinpath(path_name)
                if os.path.isfile(cachefile):
                    with open(cachefile, "rb") as f:
                        paths.extend(pickle.load(f))
                else:
                    files = self.gather_data(path)
                    with open(cachefile, "wb") as f:
                        pickle.dump(files, f)
                    paths.extend(files)
            else:
                files = self.gather_data(path)
                paths.extend(files)
        return paths


class SubSetGenerator:
    """ This class is responsible for creating and loading test and validation splits.
    Given a set of filenames, it will load a subset of samples and return unused files.

    Args:
        load_data (function): A function that can load a list of samples given a filename 
            MUST return the following format:
            [(data_1, label_1), ... , (data_n, label_n)]
        subset_name (str): The name of this subset
        num_samples (int): The number of samples in this subset
        load_path (Path): A path for loading existing splits
        save_path (Path): A path for saving the used splits
    """
    def __init__(self, load_data, subset_name, num_samples, load_path=None, save_path=None):
        self.logger = logging.getLogger(__name__)
        self.load_data = load_data
        self.num_samples = num_samples

        self.save_dir = save_path
        self.load_dir = load_path

        filename = f"{subset_name}.p"
        self.load_file = None
        if load_path is not None:
            self.load_file = Path(load_path) / filename
        self.save_file = None
        if save_path is not None:
            save_path = Path(save_path)
            if save_path.is_dir():
                self.save_file = save_path / filename
            else:
                self.logger.warning(f"save_path {save_path} is not a directory, the {subset_name} split wont be saved!")

        self.subset_name = subset_name
        self.samples = None
        self.used_filenames = None

    def _list_difference(self, a, b):
        bset = set(b)
        return [ai for ai in a if ai not in bset]

    def _load_sub_set_from_files(self, file_paths):
        # TODO: Once we remove the old pipeline, we could refactor this to return batches instead of samples
        # This would allow for a more streamlined usage and cleaner code in the GenericTrainer
        self.logger.info(f"Loading samples for {self.subset_name}")
        sample_iterator = FileSetIterator(file_paths, self.load_data)
        try:
            subset = [next(sample_iterator) for _ in range(self.num_samples)]
        except StopIteration:
            raise ValueError(f"Not enough samples to create subset {self.subset_name}")

        return subset, sample_iterator.get_remaining_files()

    def prepare_subset(self, file_paths):
        if self.load_file is not None and self.load_file.is_file():
            with open(self.load_file, 'rb') as f:
                self.used_filenames = [Path(fn) for fn in pickle.load(f)]
                unused_files = self._list_difference(file_paths, self.used_filenames)
        else:
            paths_copy = list(file_paths)
            random.shuffle(paths_copy)
            self.samples, unused_files = self._load_sub_set_from_files(paths_copy)
            self.used_filenames = self._list_difference(file_paths, unused_files)
        if self.save_file is not None:
            with open(self.save_file, 'wb') as f:
                pickle.dump([str(fn) for fn in self.used_filenames], f)
        return unused_files

    def get_samples(self):
        if self.used_filenames is None:
            raise RuntimeError(f"Cannot get subset samples without preparing files first! "
                               f"Call {type(self).__name__}.prepare_subset first.")
        if self.samples is None:  # Use this as a sort of lazy property
            self.samples, _ = self._load_sub_set_from_files(self.used_filenames)
        return self.samples
