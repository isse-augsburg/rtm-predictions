import logging
import pickle

import torch

from .TorchDataGeneratorUtils.looping_strategies import (
    LoopingStrategy, DataLoaderListLoopingStrategy, stack_aux_dicts, split_aux_dicts
)
from .TorchDataGeneratorUtils.torch_internal import FileDiscovery, FileSetIterable, CachingMode, SubSetGenerator

stack_aux_dicts = stack_aux_dicts
split_aux_dicts = split_aux_dicts


class LoopingDataGenerator:
    """ An iterable for a batches of samples stored in files.

    Args:
        data_paths (list of paths): The data paths for gathering data
        gather_data (function): A callable that gathers files given a single root directory.
            data_gather.get_filelist_within_folder is usually used for this.
        load_data (function): A function that can load a list of samples given a filename 
            MUST return the following format:
            [(data_1, label_1), ... , (data_n, label_n)]
        batch_size (int): The batch size
        num_validation_samples (int): The number of samples in the validation subset
        num_test_samples (int): The number of samples for the test subset
        split_load_path  (int): The directory to load validation and test set splits from
        split_save_path  (int): The directory to save validation and test set splits to
        num_workers (int): The number of worker processes for the dataloader. Defaults to 0 so that no additional
            processes are spawned.
        cache_path (Path): The cache directory for file lists and samples
        cache_mode (CachingMode): The cache mode. If set to FileLists, only lists of gathered files will be stored.
        looping_strategy (LoopingStrategy): The strategy for looping samples.
            Defaults to the DataLoaderListLoopingStrategy. You may want to use the NoOpLoopingStrategy if you only
            need a single epoch.
        save_torch_dataset_path (Path): Saves the Dataset to this Path. Use a full path, including a filename. Note that
            this should only be used with the DataLoaderListLoopingStrategy.
        load_torch_dataset_path (Path): Load a saved Dataset from this Path. This can improve loading times in the
            first epoch. Note that this should only be used with the DataLoaderListLoopingStrategy.
    """

    def __init__(self,
                 data_paths,
                 gather_data,
                 load_data,
                 batch_size=1,
                 num_validation_samples=0,
                 num_test_samples=0,
                 split_load_path=None,
                 split_save_path=None,
                 num_workers=0,
                 cache_path=None,
                 cache_mode=CachingMode.Both,
                 looping_strategy: LoopingStrategy = None,
                 save_torch_dataset_path=None,
                 load_torch_dataset_path=None,
                 dont_care_num_samples=False,
                 test_mode=False,
                 sampler=None,
                 load_test_set_in_training_mode=False
                 ):
        self.logger = logging.getLogger(__name__)

        self.data_paths = data_paths
        self.gather_data = gather_data
        self.load_data = load_data

        self.num_validation_samples = num_validation_samples
        self.num_test_samples = num_test_samples
        self.split_load_path = split_load_path
        self.split_save_path = split_save_path

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.store_samples = True
        self.cache_path = cache_path
        self.cache_mode = cache_mode
        self.test_mode = test_mode

        self.load_torch_dataset_path = load_torch_dataset_path
        self.save_torch_dataset_path = save_torch_dataset_path
        self.saved = False
        self.loaded = False
        self.saved_test_samples = None
        self.saved_val_samples = None

        if looping_strategy is None:
            looping_strategy = DataLoaderListLoopingStrategy(batch_size, sampler=sampler)
        self.looping_strategy = looping_strategy
        self.first = True
        if len(self.looping_strategy) > 0:
            self.first = False
        self.val_set_generator = None
        self.test_set_generator = None
        self.file_iterable = None

        self.loaded_train_set = False
        self.loaded_val_set = False
        self.loaded_test_set = False

        self.dont_care_num_samples = dont_care_num_samples

        self.try_loading_torch_datasets(load_test_set_in_training_mode)

        if self.test_mode and self.loaded_test_set:
            self.logger.info(f"Running in test mode and loaded test data set.")
        elif not self.test_mode and self.loaded_train_set and self.loaded_val_set:
            self.logger.info(f"Loaded training and validation data sets from tensors.")
            self.loaded = True
            self.saved = True
        else:
            self.load_datasets()

    def try_loading_torch_datasets(self, load_test_set_in_training_mode):
        if self.load_torch_dataset_path is None:
            return
        if self.test_mode or load_test_set_in_training_mode:
            if (self.load_torch_dataset_path / "test_set_torch.p").is_file():
                self.logger.info(f"Loading test set - torch - from {self.load_torch_dataset_path}.")
                self.saved_test_samples = torch.load(self.load_torch_dataset_path / "test_set_torch.p")
                self.loaded_test_set = True
                self.logger.info(f"Done.")
                with open(self.split_save_path / "test_set.p", "wb") as f:
                    pickle.dump(sorted(list(set([x[2]["sourcefile"] for x in self.saved_test_samples]))), f)
            if not load_test_set_in_training_mode:
                return
        if (self.load_torch_dataset_path / "train_set_torch.p").is_file():
            self.logger.info(f"Loading training set - torch - from {self.load_torch_dataset_path}.")
            self.looping_strategy.load_content(self.load_torch_dataset_path / "train_set_torch.p")
            self.loaded_train_set = True
            self.logger.info(f"Done.")
            with open(self.split_save_path / "training_set.p", "wb") as f:
                pickle.dump(sorted(list(set([x[2]["sourcefile"] for x in self.looping_strategy]))), f)
        if (self.load_torch_dataset_path / "val_set_torch.p").is_file():
            self.logger.info(f"Loading validation set - torch - from {self.load_torch_dataset_path}.")
            self.saved_val_samples = torch.load(self.load_torch_dataset_path / "val_set_torch.p")
            self.loaded_val_set = True
            self.logger.info(f"Done.")
            with open(self.split_save_path / "validation_set.p", "wb") as f:
                pickle.dump(sorted(list(set([x[2]["sourcefile"] for x in self.saved_val_samples]))), f)

    def load_datasets(self):
        self.logger.debug(f"Using {type(self.looping_strategy).__name__} for looping samples across epochs.")

        self.logger.info(f"Collecting files from {len(self.data_paths)} directories")
        all_files = self._discover_files(self.data_paths, self.gather_data)

        self.logger.info("Getting validation and test data splits.")
        # if self.saved_val_samples is None or self.saved_test_samples is None:
        self.val_set_generator = SubSetGenerator(self.load_data, "validation_set", self.num_validation_samples,
                                                 load_path=self.split_load_path, save_path=self.split_save_path,
                                                 dont_care_num_samples=self.dont_care_num_samples)
        self.test_set_generator = SubSetGenerator(self.load_data, "test_set", self.num_test_samples,
                                                  load_path=self.split_load_path, save_path=self.split_save_path,
                                                  dont_care_num_samples=self.dont_care_num_samples)
        remaining_files = self.val_set_generator.prepare_subset(all_files)
        remaining_files = self.test_set_generator.prepare_subset(remaining_files)
        if self.split_save_path is not None:
            filename = self.split_save_path / "training_set.p"
            with open(filename, 'wb') as f:
                pickle.dump([str(fn) for fn in remaining_files], f)
        self.logger.info(f"{len(remaining_files)} remaining files will be loaded using {self.num_workers} workers.")
        self.file_iterable = FileSetIterable(remaining_files, self.load_data,
                                             cache_path=self.cache_path, cache_mode=self.cache_mode)
        self.logger.info("Data generator initialization is done.")

    def _discover_files(self, data_paths, gather_data):
        self.logger.info(f"Gathering files from {len(data_paths)} paths...")
        data_paths = [str(x) for x in data_paths]
        discovery = FileDiscovery(gather_data, cache_path=self.cache_path, cache_mode=self.cache_mode)
        paths = discovery.discover(data_paths)
        self.logger.debug(f"Gathered {len(paths)} files.")
        return paths

    def __iter__(self):
        if self.first and not self.loaded_train_set:
            # By choosing drop_last=False we may get up to num_workers*(batch_size-1) short batches in the first epoch.
            # The behaviour in the second depends on the used LoopingStrategy, but by default we will only see one short
            # sample in the following epochs
            dataloader = torch.utils.data.DataLoader(self.file_iterable, drop_last=False,
                                                     batch_size=self.batch_size, num_workers=self.num_workers)

            def store_batch(batch):
                batch = [e.clone() for e in batch[:2]] + batch[2:]
                self.looping_strategy.store(batch)
                return batch

            iterator = map(store_batch, dataloader)
            self.first = False
        else:
            if not self.saved and self.save_torch_dataset_path is not None:
                if not (self.save_torch_dataset_path / "train_set_torch.p").is_file():
                    self.looping_strategy.dump_content(self.save_torch_dataset_path / "train_set_torch.p")
                self.saved = True

            iterator = self.looping_strategy.get_new_iterator()

        return map(tuple, iterator)

    def __len__(self):
        return len(self.looping_strategy)

    def get_validation_samples(self):
        """ Get the set of validation samples
        """
        if self.saved_val_samples is None:
            self.saved_val_samples = self.val_set_generator.get_samples()
            if self.save_torch_dataset_path is not None:
                torch.save(self.saved_val_samples, self.save_torch_dataset_path / "val_set_torch.p")
        return self.saved_val_samples

    def get_test_samples(self):
        """ Get the set of test samples
        """
        if self.saved_test_samples is None:
            self.saved_test_samples = self.test_set_generator.get_samples()
            if self.save_torch_dataset_path is not None:
                torch.save(self.saved_test_samples, self.save_torch_dataset_path / "test_set_torch.p")
        return self.saved_test_samples
