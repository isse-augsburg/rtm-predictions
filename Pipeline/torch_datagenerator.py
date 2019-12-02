import logging
import math
import os
import pickle
import random
from abc import abstractmethod, ABC
from enum import Enum
from pathlib import Path
from queue import Queue

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


class LoopingStrategy(ABC):
    """ LoopingStrategies are used to repeat samples after the first epoch.
    The LoopingDataGenerator will pass every loaded batch into the store function.
    Once a sample iterator is exhausted, a new one will be created using the
    get_new_iterator function.
    """
    def __init__(self):
        pass

    def store(self, batch):
        """ Store a new sample into the strategies buffer

        Args:
            batch (tuple of feature-batch and label-batch)
        """
        pass

    @abstractmethod
    def get_new_iterator(self):
        """ This should return a new iterator.
        The new iterator must yield all samples that were previously stored using store.
        Yielded objects should be in the same batch form store expects.
        Also, the strategy is responsible for shuffling samples.
        """
        pass


class SimpleListLoopingStrategy(LoopingStrategy):
    """ This strategy just stores batches in a list and shuffles that list between epochs.
    This strategy is really fast, but shuffling on a batch basis instead of samples
    reduces training performance and overall training results.
    """
    def __init__(self):
        super().__init__()
        self.batches = []

    def store(self, batch):
        self.batches.append(batch)

    def get_new_iterator(self):
        random.shuffle(self.batches)
        return iter(self.batches)


class ComplexListLoopingStrategy(LoopingStrategy):
    """ This strategy stores individual samples and shuffles these between epochs.
    This is pretty slow compared to the SimpleListLoopingStrategy, but it gives
    better results in training.
    """
    def __init__(self, batch_size):
        super().__init__()
        self.features = []
        self.labels   = []
        self.batch_size = batch_size

    def store(self, batch):
        features, labels = batch
        self.features.extend(torch.split(features, 1))
        self.labels.extend(torch.split(labels, 1))

    def get_new_iterator(self):
        samples = list(zip(self.features, self.labels))
        random.shuffle(samples)
        list_iter = iter(samples)
        while True:
            try:
                batch = [next(list_iter) for _ in range(self.batch_size)]
            except StopIteration:
                break
            features = [b[0].squeeze(0) for b in batch]
            labels = [b[1].squeeze(0) for b in batch]
            yield torch.stack(features), torch.stack(labels)


class DataLoaderListLoopingStrategy(LoopingStrategy, torch.utils.data.Dataset):
    """ This strategy shuffles on a sample basis like the ComplexListLoopingStrategy,
    but it relies on the torch DataLoader for shuffling.
    It seems to have slightly better performance than the ComplexList approach.
    """
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.batches = []

    def store(self, batch):
        self.batches.append(batch)

    def get_new_iterator(self):
        return iter(torch.utils.data.DataLoader(self, shuffle=True, batch_size=self.batch_size))

    def __len__(self):
        return len(self.batches) * self.batch_size

    def __getitem__(self, index):
        batch_index = int(index / self.batch_size)
        subindex = index - batch_index * self.batch_size
        features, labels = self.batches[batch_index]
        return features[subindex], labels[subindex]


class NoOpLoopingStrategy(LoopingStrategy):
    """ A "do-nothing" strategy that will just forget everything stored in it.
    This is automatically used if you only run a single epoch and will prevent
    the huge memory requirements that the other strategies have.
    """
    def __init__(self):
        super().__init__()

    def get_new_iterator(self):
        return iter([])


class SubSetGenerator:
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


class LoopingDataGenerator:
    def __init__(self,
                 data_paths,
                 gather_data,
                 load_data,
                 batch_size=1,
                 epochs=1,
                 num_validation_samples=0,
                 num_test_samples=0,
                 split_load_path=None,
                 split_save_path=None,
                 num_workers=0,
                 cache_path=None,
                 cache_mode=CachingMode.Both,
                 looping_strategy: LoopingStrategy = None
                 ):
        self.epochs = epochs  # For compatibility with the MasterTrainer
        self.batch_size = batch_size  # For compatibility with the MasterTrainer
        self.remaining_epochs = epochs
        self.store_samples = True
        self.batch_size = batch_size
        self.cache_path = cache_path
        self.cache_mode = cache_mode
        self.logger = logging.getLogger(__name__)

        if looping_strategy is None:
            if epochs > 1:
                # FIXME: Default was DataLoaderListLoopingStrategy, but currently produces wrong at least in
                #  validation set. Labels are all 1. Anyway, the ComplexListLoopingStrategy seems to work fine and fast
                looping_strategy = ComplexListLoopingStrategy(batch_size)
            else:
                looping_strategy = NoOpLoopingStrategy()
        self.looping_strategy = looping_strategy
        self.logger.debug(f"Using {type(self.looping_strategy).__name__} for looping samples across epochs.")

        all_files = self._discover_files(data_paths, gather_data)
        self.logger.info("Generating validation and test data splits.")
        self.eval_set_generator = SubSetGenerator(load_data, "validation_set", num_validation_samples,
                                                  load_path=split_load_path, save_path=split_save_path)
        self.test_set_generator = SubSetGenerator(load_data, "test_set", num_test_samples,
                                                  load_path=split_load_path, save_path=split_save_path)
        remaining_files = self.eval_set_generator.prepare_subset(all_files)
        remaining_files = self.test_set_generator.prepare_subset(remaining_files)
        self.logger.info(f"{len(remaining_files)} files remain after splitting eval and test sets.")

        loader_iterable = FileSetIterable(remaining_files, load_data,
                                          cache_path=cache_path, cache_mode=cache_mode)
        self.iterator = iter(torch.utils.data.DataLoader(loader_iterable,
                                                         batch_size=batch_size, num_workers=num_workers))
        self.logger.info("Data generator initialization is done.")

    def _discover_files(self, data_paths, gather_data):
        self.logger.info(f"Gathering files from {len(data_paths)} paths...")
        data_paths = [str(x) for x in data_paths]
        discovery = FileDiscovery(gather_data, cache_path=self.cache_path, cache_mode=self.cache_mode)
        paths = discovery.discover(data_paths)
        self.logger.debug(f"Gathered {len(paths)} files.")
        return paths

    def __iter__(self):
        return self

    def __next__(self):
        try:
            batch = next(self.iterator)
            if self.store_samples:
                batch = [e.clone() for e in batch]
                self.looping_strategy.store(batch)
        except StopIteration:
            self.remaining_epochs -= 1
            if self.remaining_epochs == 0:
                raise StopIteration
            self.logger.info(f"Starting epoch {self.epochs - self.remaining_epochs + 1}")
            self.store_samples = False
            self.iterator = self.looping_strategy.get_new_iterator()
            batch = next(self.iterator)
        return batch[0], batch[1]

    def get_validation_samples(self):
        return self.eval_set_generator.get_samples()

    def get_test_samples(self):
        return self.test_set_generator.get_samples()

    def end_threads(self):
        # TODO: Dummy method for compatibility with the old pipeline. Remove this once the old pipeline
        # is removed
        pass

    def get_current_queue_length(self):
        # TODO: Dummy method for compatibility with the old pipeline. Remove this once the old pipeline
        # is removed
        return "unk"
