# Copyright © 2021 Chris Hughes
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Iterable


class RunHistory(ABC):
    """
    The abstract base class which defines the API for a :class:`~pytorch_accelerated.trainer.Trainer`'s run history.
    """

    @abstractmethod
    def get_metric_names(self) -> Iterable:
        """
        Return a set containing of all unique metric names which are being tracked.

        :return: an iterable of the unique metric names
        """
        pass

    @abstractmethod
    def get_metric_values(self, metric_name) -> Iterable:
        """
        Return all of the values that have been recorded for the given metric.

        :param metric_name: the name of the metric being tracked
        :return: an ordered iterable of values that have been recorded for that metric
        """
        pass

    @abstractmethod
    def get_latest_metric(self, metric_name):
        """
        Return the most recent value that has been recorded for the given metric.

        :param metric_name: the name of the metric being tracked
        :return: the last recorded value
        """
        pass

    @abstractmethod
    def set_metric_name_prefix(self, prefix=""):
        """
        Set a prefix which will be prepended to any metric name which is tracked.

        :param prefix: a prefix which will be prepended to any metric name which is tracked
        """
        pass

    @property
    @abstractmethod
    def metric_name_prefix(self):
        """
        :return: the prefix which wil be prepended to any metric name
        """
        pass

    @abstractmethod
    def update_metric(self, metric_name, metric_value):
        """
        Record the value for the given metric.

        :param metric_name: the name of the metric being tracked
        :param metric_value: the value to record
        """
        pass

    @abstractmethod
    def delete_metric(self, metric_name):
        """
        Delete the metric from the run history.

        :param metric_name: the name of the metric to delete
        """
        pass

    @property
    @abstractmethod
    def current_epoch(self) -> int:
        """
        Return the value of the current epoch.

        :return: an int representing the value of the current epoch
        """
        pass

    @abstractmethod
    def _increment_epoch(self):
        """
        Increment the value of the current epoch
        """
        pass

    @abstractmethod
    def reset(self):
        """
        Reset the state of the :class:`RunHistory`
        """
        pass


class InMemoryRunHistory(RunHistory):
    """
    An implementation of :class:`RunHistory` which stores all recorded values in memory.
    """

    def __init__(self):
        self._current_epoch = 1
        self._metrics = defaultdict(list)
        self._prefix = ""

    def get_metric_names(self):
        return set(self._metrics.keys())

    def get_metric_values(self, metric_name):
        return self._metrics[metric_name]

    def get_latest_metric(self, metric_name):
        if len(self._metrics[metric_name]) > 0:
            return self._metrics[metric_name][-1]
        else:
            raise ValueError(
                f"No values have been recorded for the metric {metric_name}"
            )

    def update_metric(self, metric_name, metric_value):
        self._metrics[f"{self._prefix}{metric_name}"].append(metric_value)

    def set_metric_name_prefix(self, prefix=""):
        self._prefix = prefix

    def delete_metric(self, metric_name):
        del self._metrics[metric_name]

    @property
    def metric_name_prefix(self):
        return self._prefix

    @property
    def current_epoch(self):
        return self._current_epoch

    def _increment_epoch(self):
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
