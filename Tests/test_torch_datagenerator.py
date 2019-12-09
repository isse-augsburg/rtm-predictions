import pickle
import shutil
import unittest
from pathlib import Path
import tempfile
import random

import h5py
import numpy as np
import torch

import Tests.resources_for_testing as resources
import Pipeline.Utils.torch_internal as ti
import Pipeline.Utils.looping_strategies as ls
import Pipeline.torch_datagenerator as td
import Pipeline.data_gather as dg


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


class TorchTestCase(unittest.TestCase):
    def assertListOfSamplesEqual(self, l1, l2):
        for s1, s2 in zip(l1, l2):
            self.assertTrue(s1[0].equal(s2[0]))
            self.assertTrue(s1[1].equal(s2[1]))


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


def _dummy_dataloader_fn(filename):
    with h5py.File(filename, "r") as f:
        return [(list(str(filename).encode("utf-8")), np.array([i, 0]))
                for i in range(len(f["post"]["singlestate"].keys()))]


class TestFileSetIterator(TorchTestCase):
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
            orig_samples = list(iterator)
            iterator = ti.FileSetIterator(list(self.test_set.erf_files), None, cache_path=cache_path)
            cached_samples = list(iterator) 
            self.assertListOfSamplesEqual(orig_samples, cached_samples)

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


class TestSubsetGenerator(TorchTestCase):
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
            save_unused_files = subset_gen.prepare_subset(self.test_set.erf_files)
            save_samples = subset_gen.get_samples()

            subset_gen = ti.SubSetGenerator(_dummy_dataloader_fn, "test_datasplit", self.num_split_samples,
                                            load_path=splitpath)
            shuffled_files = list(self.test_set.erf_files)
            random.shuffle(shuffled_files)
            load_unused_files = subset_gen.prepare_subset(shuffled_files)
            load_samples = subset_gen.get_samples()

            self.assertCountEqual(save_unused_files, load_unused_files)
            self.assertListOfSamplesEqual(save_samples, load_samples)


class TestLoopingStrategies(TorchTestCase):
    def setUp(self):
        self.test_set = _TestSetInfo()
        self.batch_size = 1
        self.looping_strategies = [lambda: ls.SimpleListLoopingStrategy(),
                                   lambda: ls.ComplexListLoopingStrategy(self.batch_size),
                                   lambda: ls.DataLoaderListLoopingStrategy(self.batch_size)]

    def test_strategies(self):
        for strategyfn in self.looping_strategies:
            strategy = strategyfn()
            with self.subTest(msg=f"Checking stragety {type(strategy).__name__}"):
                dataloader = td.LoopingDataGenerator(self.test_set.paths, dg.get_filelist_within_folder,
                                                     _dummy_dataloader_fn, looping_strategy=strategy, epochs=2)
                first_epoch = set((b[0][0], b[1][0]) for b, _ in zip(dataloader, range(self.test_set.num_samples)))
                second_epoch = set((b[0][0], b[1][0]) for b in dataloader)
                # TODO: This will probably not work 
                self.assertSetEqual(first_epoch, second_epoch)
