from abc import ABC, abstractmethod
import random

import torch


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
        self.labels = []
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
        self.features = []
        self.labels = []

    def store(self, batch):
        features, labels = batch
        self.features.extend(f.squeeze(0) for f in torch.split(features, 1))
        self.labels.extend(l.squeeze(0) for l in torch.split(labels, 1))

    def get_new_iterator(self):
        return iter(torch.utils.data.DataLoader(self, shuffle=True, batch_size=self.batch_size))

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]


class NoOpLoopingStrategy(LoopingStrategy):
    """ A "do-nothing" strategy that will just forget everything stored in it.
    This is automatically used if you only run a single epoch and will prevent
    the huge memory requirements that the other strategies have.
    """
    def __init__(self):
        super().__init__()

    def get_new_iterator(self):
        return iter([])
