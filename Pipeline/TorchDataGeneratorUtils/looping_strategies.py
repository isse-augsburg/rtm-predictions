import random
from abc import ABC, abstractmethod

import torch


def split_aux_dicts(aux):
    for i in range(len(aux[list(aux)[0]])):
        yield {
            key: aux[key][i]
            for key in aux.keys()
        }


def stack_aux_dicts(auxs):
    return {
        key: [aux[key] for aux in auxs]
        for key in auxs[0].keys()
    }


class LoopingStrategy(ABC):
    """ LoopingStrategies are used to repeat samples after the first epoch.
    The LoopingDataGenerator will pass every loaded batch into the store function.
    Once a sample iterator is exhausted, a new one will be created using the
    get_new_iterator function.
    """

    def __init__(self):
        self.num_samples = 0
        pass

    def store(self, batch):
        """ Store a new sample into the strategies buffer

        Args:
            batch (tuple of feature-batch and label-batch)
        """
        self.num_samples += len(batch[0])

    def __len__(self):
        return self.num_samples

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
        super().store(batch)
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
        self.labels = []
        self.aux = []
        self.batch_size = batch_size

    def store(self, batch):
        super().store(batch)
        features, labels, aux = batch
        self.features.extend(torch.split(features, 1))
        self.labels.extend(torch.split(labels, 1))
        self.aux.extend(split_aux_dicts(aux))

    def get_new_iterator(self):
        samples = list(zip(self.features, self.labels, self.aux))
        random.shuffle(samples)
        list_iter = iter(samples)
        while True:
            try:
                batch = [next(list_iter) for _ in range(self.batch_size)]
            except StopIteration:
                break
            features = [b[0].squeeze(0) for b in batch]
            labels = [b[1].squeeze(0) for b in batch]
            auxs = [b[2] for b in batch]
            yield torch.stack(features), torch.stack(labels), stack_aux_dicts(auxs)


class DataLoaderListLoopingStrategy(LoopingStrategy, torch.utils.data.Dataset):
    """ This strategy shuffles on a sample basis like the ComplexListLoopingStrategy,
    but it relies on the torch DataLoader for shuffling.
    It seems to have slightly better performance than the ComplexList approach.
    """

    def __init__(self, batch_size, sampler=None):
        super().__init__()
        self.sampler = sampler
        self.batch_size = batch_size
        self.features = []
        self.labels = []
        self.aux = []

    def store(self, batch):
        super().store(batch)
        features, labels, aux = batch
        self.features.extend(f.squeeze(0) for f in torch.split(features, 1))
        self.labels.extend(l.squeeze(0) for l in torch.split(labels, 1))
        self.aux.extend(split_aux_dicts(aux))

    def get_new_iterator(self):
        if not hasattr(self, "sampler") or self.sampler is None:
            return iter(torch.utils.data.DataLoader(self, shuffle=True, batch_size=self.batch_size))
        else:
            sampler = self.sampler((self.features, self.labels, self.aux))
            return iter(torch.utils.data.DataLoader(self, sampler=sampler, batch_size=self.batch_size))

    def __getitem__(self, index):
        return self.features[index], self.labels[index], self.aux[index]


class NoOpLoopingStrategy(LoopingStrategy):
    """ A "do-nothing" strategy that will just forget everything stored in it.
    This is automatically used if you only run a single epoch and will prevent
    the huge memory requirements that the other strategies have.
    """

    def __init__(self):
        super().__init__()

    def get_new_iterator(self):
        return iter([])

    def __len__(self):
        return 0
