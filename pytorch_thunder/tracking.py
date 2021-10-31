from abc import ABC, abstractmethod
from collections import defaultdict


class RunHistory(ABC):
    @abstractmethod
    def get_metric_names(self):
        pass

    @abstractmethod
    def get_metric(self, metric_name):
        pass

    @abstractmethod
    def get_latest_metric(self, metric_name):
        pass

    @abstractmethod
    def update_metric(self, metric_name, metric_value):
        pass

    @property
    @abstractmethod
    def current_epoch(self):
        pass

    @abstractmethod
    def increment_epoch(self):
        pass


class InMemoryRunHistory(RunHistory):
    def __init__(self):
        self._current_epoch = 0
        self._metrics = defaultdict(list)

    def get_metric_names(self):
        return set(self._metrics.keys())

    def get_metric(self, metric_name):
        return self._metrics[metric_name]

    def get_latest_metric(self, metric_name):
        if len(self._metrics[metric_name]) > 0:
            return self._metrics[metric_name][-1]

    def update_metric(self, metric_name, metric_value):
        self._metrics[metric_name].append(metric_value)

    @property
    def current_epoch(self):
        return self._current_epoch

    def increment_epoch(self):
        self._current_epoch += 1


class AverageMeter:
    """Computes and stores the average and current value
    Taken from timm"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
