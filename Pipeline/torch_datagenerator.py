import json
import logging
import math
import os
import pickle
import random
import socket
import threading
from enum import Enum
from pathlib import Path
from time import sleep, time
from queue import Queue

import numpy as np
import torch

from Pipeline import data_gather as dg, data_loader_sensor as dls


class HDF5Iterator:
    def __init__(self, files, dataloader, cache_path, worker_id):
        self.files = files
        self.data_function = dataloader
        self.cache_path = cache_path
        self.sample_queue = Queue()
        self.worker_id = worker_id

    def get_cache_path_for_file(self, filename):
        s_path = Path(filename)
        s_path = self.cache_path.joinpath(s_path.stem)
        return s_path

    def load_cached_samples(self, filename):
        if self.cache_path is not None:
            s_path = self.get_cache_path_for_file(filename)
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

    def transform_to_tensor_and_cache(self, i, num, s_path):
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

    def load_file(self):
        while True:
            if len(self.files) == 0:
                return False
            fn = self.files.pop(0)
            print(f"Worker {self.worker_id} loading file {fn}")
            if self.load_cached_samples(fn):
                break # This file was already cached; nothing to do here

            instance = self.data_function(fn)
            if instance is None:
                continue
            else:
                # TODO: assert_instance_correctness
                s_path = None
                if self.cache_path is not None:
                    s_path = self.get_cache_path_for_file(fn)
                    s_path.mkdir(parents=True, exist_ok=True)
                for num, i in enumerate(instance):
                    self.transform_to_tensor_and_cache(i, num, s_path)
                break
        return True

    def __next__(self):
        if self.sample_queue.empty():
            if not self.load_file():
                raise StopIteration
        return self.sample_queue.get()


class CachingMode(Enum):
    Nothing = 1
    Both = 2
    FileList = 3


class FileDiscovery:
    def __init__(self, gather_data, cache_path=None, cache_mode=CachingMode.FileList):
        self.filelist_cache_path = None
        if cache_path is not None and cache_mode in [CachingMode.Both, CachingMode.FileList]:
            self.filelist_cache_path = Path(cache_path).joinpath("filelists")
            self.filelist_cache_path.mkdir(parents=True, exist_ok=True)
        self.gather_data = gather_data

    def discover(self, data_paths):
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


class DataLoaderIterable(torch.utils.data.IterableDataset):
    def __init__(self, data_paths, gather_data, load_data, cache_path=None, cache_mode=CachingMode.Both):
        self.data_paths = [str(x) for x in data_paths]
        self.cache_path = cache_path
        self.paths = []
        self.gather_data = gather_data
        self.load_data = load_data

        self.logger = logging.getLogger(__name__)
        discovery = FileDiscovery(gather_data, cache_path=cache_path, cache_mode=cache_mode)
        self.paths = discovery.discover(self.data_paths)
        print(f"Gathered {len(self.paths)} files")
        self.sample_cache_path = None
        if cache_path is not None and cache_mode in [CachingMode.Both]:
            self.sample_cache_path = Path(cache_path).joinpath(self.load_data.__name__)
            self.sample_cache_path.mkdir(parents=True, exist_ok=True)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = 0
        if worker_info is None:  # single-process data loading, return the full iterator
            worker_paths = self.paths
        else:  # in a worker process
            # split workload
            per_worker = int(math.ceil(len(self.paths) / float(worker_info.num_workers)))
            print(f"Each worker will process {per_worker} files!")
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = iter_start + per_worker
            worker_paths = self.paths[iter_start:iter_end]
        return HDF5Iterator(worker_paths, self.load_data, self.sample_cache_path, worker_id)


class LoopingDataGenerator:
    def __init__(self,
                 data_paths,
                 gather_data,
                 load_data,
                 batch_size=1,
                 epochs=1,
                 num_workers=0,
                 cache_path=None,
                 cache_mode=CachingMode.Both,
                 ):
        loader_iterable = DataLoaderIterable(data_paths, gather_data, load_data,
                                             cache_path=cache_path, cache_mode=cache_mode)
        self.iterator = iter(torch.utils.data.DataLoader(loader_iterable, batch_size=1, num_workers=num_workers))
        self.remaining_epochs = epochs
        self.samples = []
        self.store_samples = True
        self.batch_size = batch_size

    def __iter__(self):
        return self

    def __next__(self):
        try:
            samples = [next(self.iterator) for _ in range(self.batch_size)]
            samples = [[e.clone() for e in sample] for sample in samples]
            if self.store_samples:
                self.samples.extend(samples)
        except StopIteration:
            self.store_samples = False
            random.shuffle(self.samples)
            self.iterator = iter(self.samples)
            samples = [next(self.iterator) for _ in range(self.batch_size)]
        data = [i[0] for i in samples]
        labels = [i[1] for i in samples]
        batch_data = torch.stack(data)
        batch_labels = torch.stack(labels)
        return batch_data, batch_labels
