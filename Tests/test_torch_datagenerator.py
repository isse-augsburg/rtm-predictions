import logging
import random
import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import torch

import Pipeline.TorchDataGeneratorUtils.looping_strategies as ls
import Pipeline.TorchDataGeneratorUtils.torch_internal as ti
import Pipeline.data_gather as dg
import Pipeline.torch_datagenerator as td
import Resources.testing as resources
from Utils.natural_sorting import natural_sort_key

logger = logging.getLogger()
logger.level = logging.ERROR
stream_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stream_handler)


class _TestSetInfo:
    def __init__(self, run_name="2019-07-11_15-14-48", num_runs=100):
        self.run_name = run_name
        self.p = resources.test_pipeline_dir / f"{self.run_name}_{num_runs}p"
        self.paths = [self.p]
        self.erf_files = [self.p / f'{i}/{self.run_name}_{i}_RESULT.erfh5' for i in range(100)]
        self._num_samples = None

    @property
    def num_samples(self):
        if self._num_samples is None:
            self._num_samples = 0
            for f in self.erf_files:
                with h5py.File(f, "r") as hdf:
                    self._num_samples += len(hdf["post"]["singlestate"].keys())

        return self._num_samples


def _dummy_dataloader_fn(filename):
    """ This is a dummy data loader that returns a fake sample for every singlestate in the given file.
    The data will always be the filename while the label is the index of the sample
    """
    def fn_to_list(fn):
        lst = list(str(filename).encode("utf-8"))
        return lst + (200 - len(lst)) * [0]
    with h5py.File(filename, "r") as f:
        return [(fn_to_list(f), np.array([i, 0]))
                for i in range(len(f["post"]["singlestate"].keys()))]


class SampleWrapper:
    """ A wrapper around samples returned using the _dummy_dataloader_fn
    Provides a by-value hash function and a nice __repr__ method
    """
    def __init__(self, sample):
        self.data, self.label = sample

    def __eq__(self, other):
        od, ol = other.data, other.label
        if od.shape != self.data.shape or ol.shape != self.label.shape:
            return False
        return od.allclose(self.data) and ol.allclose(self.label)

    def to_list(self, tensor):
        return tuple(x.item() for x in tensor)

    def __hash__(self):
        return hash((self.to_list(self.data), self.to_list(self.label)))

    def __repr__(self):
        fn = bytes(list(int(t.item()) for t in self.data)).decode("utf-8")
        idx = int(self.label[0].item())
        return f"Sample {fn} - {idx}"


class TestFileDiscovery(unittest.TestCase):
    def setUp(self):
        self.test_set = _TestSetInfo()

    def test_gathering(self):
        discovery = ti.FileDiscovery(dg.get_filelist_within_folder)
        discoverd_files = discovery.discover(self.test_set.paths)
        self.assertEqual(sorted(discoverd_files), sorted(self.test_set.erf_files))

    def test_caching(self):
        with tempfile.TemporaryDirectory(prefix="FileDiscovery_Cache") as cache_path:
            discovery = ti.FileDiscovery(dg.get_filelist_within_folder, cache_path=cache_path)
            original_files = discovery.discover(self.test_set.paths)
            self.assertEqual(sorted(original_files), sorted(self.test_set.erf_files))
            discovery = ti.FileDiscovery(None, cache_path=cache_path)
            cached_files = discovery.discover(self.test_set.paths)
            self.assertEqual(sorted(cached_files), sorted(original_files))


class TestFileSetIterator(unittest.TestCase):
    def setUp(self):
        self.test_set = _TestSetInfo()

    def test_loading(self):
        iterator = ti.FileSetIterator(self.test_set.erf_files, _dummy_dataloader_fn)
        samples = set(iterator)
        self.assertEqual(len(samples), self.test_set.num_samples)

    def test_caching(self):
        with tempfile.TemporaryDirectory(prefix="FileSetIterator_Cache") as cache_path:
            cache_path = Path(cache_path)
            iterator = ti.FileSetIterator(list(self.test_set.erf_files), _dummy_dataloader_fn, cache_path=cache_path)
            orig_samples = list(SampleWrapper(sample) for sample in iterator)
            iterator = ti.FileSetIterator(list(self.test_set.erf_files), None, cache_path=cache_path)
            cached_samples = list(SampleWrapper(sample) for sample in iterator) 
            self.assertListEqual(orig_samples, cached_samples)

    def test_get_remaining_files(self):
        iterator = ti.FileSetIterator(list(self.test_set.erf_files), _dummy_dataloader_fn)
        next(iterator)
        self.assertEqual(iterator.get_remaining_files(), self.test_set.erf_files[1:])


class TestFileSetIterable(unittest.TestCase):
    def setUp(self):
        self.test_set = _TestSetInfo()
        self.iterable = ti.FileSetIterable(self.test_set.erf_files, _dummy_dataloader_fn)

    def test_simple(self):
        samples = set(self.iterable)
        self.assertEqual(len(samples), self.test_set.num_samples)

    def test_dataloader(self):
        loader = torch.utils.data.DataLoader(self.iterable)
        samples = set((b[0][0], b[1][0]) for b in loader)
        self.assertEqual(len(samples), self.test_set.num_samples)

    def test_dataloader_mp(self):
        loader = torch.utils.data.DataLoader(self.iterable, num_workers=8)
        samples = set((b[0][0].clone(), b[1][0].clone()) for b in loader)
        self.assertEqual(len(samples), self.test_set.num_samples)


class TestSubsetGenerator(unittest.TestCase):
    def setUp(self):
        self.test_set = _TestSetInfo()
        self.num_split_samples = 400

    def test_split_size(self):
        subset_gen = ti.SubSetGenerator(_dummy_dataloader_fn, "test_datasplit", self.num_split_samples)
        unused_files = subset_gen.prepare_subset(self.test_set.erf_files)
        self.assertCountEqual(unused_files + subset_gen.used_filenames, self.test_set.erf_files, msg="Splitting should"
                              "not loose any files")
        samples = subset_gen.get_samples()
        self.assertEqual(len(samples), self.num_split_samples) 

    def test_split_save_load(self):
        with tempfile.TemporaryDirectory(prefix="SubsetGenerator_Splits") as splitpath:
            subset_gen = ti.SubSetGenerator(_dummy_dataloader_fn, "test_datasplit", self.num_split_samples,
                                            save_path=splitpath)
            files = list(self.test_set.erf_files)
            save_unused_files = subset_gen.prepare_subset(self.test_set.erf_files)
            save_samples = subset_gen.get_samples()
            self.assertListEqual(save_unused_files, sorted(save_unused_files, key=natural_sort_key))

            subset_gen = ti.SubSetGenerator(_dummy_dataloader_fn, "test_datasplit", self.num_split_samples,
                                            load_path=splitpath)
            shuffled_files = list(self.test_set.erf_files)
            random.shuffle(shuffled_files)
            load_unused_files = subset_gen.prepare_subset(shuffled_files)
            self.assertListEqual(files, self.test_set.erf_files, "Splitting should not affect the given file list")
            load_samples = subset_gen.get_samples()

            self.assertCountEqual(save_unused_files, load_unused_files)
            # TODO: Proper file sorting will result in correct ordering of samples here
            save_samples = list(SampleWrapper(sample) for sample in save_samples) 
            load_samples = list(SampleWrapper(sample) for sample in load_samples) 

            self.assertSetEqual(set(save_samples), set(load_samples))
            self.assertListEqual(save_samples, load_samples)


class TestLoopingStrategies(unittest.TestCase):
    def setUp(self):
        self.test_set = _TestSetInfo()
        self.batch_size = 1
        self.looping_strategies = [lambda: ls.SimpleListLoopingStrategy(),
                                   lambda: ls.ComplexListLoopingStrategy(self.batch_size),
                                   lambda: ls.DataLoaderListLoopingStrategy(self.batch_size),
                                   ]

    def test_strategies(self):
        for strategyfn in self.looping_strategies:
            strategy = strategyfn()
            with self.subTest(msg=f"Checking strategy {type(strategy).__name__}"):
                dataloader = td.LoopingDataGenerator(self.test_set.paths, dg.get_filelist_within_folder,
                                                     _dummy_dataloader_fn, looping_strategy=strategy)
                self.assertEqual(len(dataloader), 0)
                first_epoch = set(SampleWrapper((b[0][i], b[1][i]))
                                  for b in dataloader
                                  for i in range(len(b[0])))
                self.assertEqual(len(dataloader), self.test_set.num_samples)
                second_epoch = set(SampleWrapper((b[0][i], b[1][i])) for b in dataloader for i in range(len(b[0])))
                self.assertSetEqual(first_epoch, second_epoch)

    def test_noop_strategy(self):
        strategy = ls.NoOpLoopingStrategy()
        dataloader = td.LoopingDataGenerator(self.test_set.paths, dg.get_filelist_within_folder,
                                             _dummy_dataloader_fn, looping_strategy=strategy)
        self.assertEqual(len(dataloader), 0)
        first_epoch = set(SampleWrapper((b[0][i], b[1][i]))
                          for b in dataloader
                          for i in range(len(b[0])))
        self.assertEqual(len(first_epoch), self.test_set.num_samples)
        self.assertEqual(len(dataloader), 0)
        second_epoch = set(SampleWrapper((b[0][i], b[1][i])) for b in dataloader for i in range(len(b[0])))
        self.assertEqual(len(second_epoch), 0)
