# Copyright © 2021 Chris Hughes
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

    @abstractmethod
    def reset(self):
        pass


class InMemoryRunHistory(RunHistory):
    def __init__(self):
        self._current_epoch = 1
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

    def reset(self):
        self._current_epoch = 1
        self._metrics = defaultdict(list)


class LossTracker:
    def __init__(self):
        self.loss_value = 0
        self._average = 0
        self.total_loss = 0
        self.running_count = 0

    def reset(self):
        self.loss_value = 0
        self._average = 0
        self.total_loss = 0
        self.running_count = 0

    def update(self, loss_batch_value, batch_size=1):
        self.loss_value = loss_batch_value
        self.total_loss += loss_batch_value * batch_size
        self.running_count += batch_size
        self._average = self.total_loss / self.running_count

    @property
    def average(self):
        return self._average
