import collections
from typing import Dict, Callable, Tuple, Union

import torch
import torchmetrics as tm


class MetricSet(collections.UserDict):
    """
    Wrapper around torchmetrics Metric and MetricCollection objects.
    It adds the ability to filter arguments, but can be used transparently as a Metric object.

    Initialization:
        - data: a dictionary of metrics, with a name and a tuple containing the torchmetrics metric object and a callback to filter arguments that are passed to it
    """

    def __init__(self, data: Dict[str, Tuple[Union[tm.Metric, tm.MetricCollection], Callable]] = dict()):
        super().__init__(data)
        # Data is set in the superclass, this is just to document the type
        self.data: Dict[str, Tuple[Union[tm.Metric, tm.MetricCollection], Callable]]
        self._device = None

    def __call__(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Compute metrics on a batch, accumulating results.

        @param args: any positional arg
        @param kwargs: any key-value arg
        @return: dictionary with resulting metric value, over the current batch only
        """
        result = dict()
        for metric_name, (metric, filter_function) in self.items():
            metric.to(self._device)
            filtered_results = filter_function(*args, **kwargs, device=self._device)
            if isinstance(filtered_results, dict):
                result[metric_name] = metric(**filtered_results)
            else:
                result[metric_name] = metric(*filtered_results)

        self.__flatten_subdicts(result)

        return result

    def compute(self) -> Dict[str, torch.Tensor]:
        """
        Compute final metrics over all batches

        @return: resulting metrics over all accumulated batches
        """
        result = dict()
        for metric_name, (metric, _) in self.items():
            metric.to(self._device)
            result[metric_name] = metric.compute()

        self.__flatten_subdicts(result)
        return result

    def reset(self, metric_name: str = None) -> None:
        """
        Reset the accumulators, of one or all metrics.
        To be called after each epoch, before starting to evaluate the first batch.

        @param metric_name: if a name is passed only that metric is reset, otherwise all metrics are reset
        """
        if metric_name is not None:
            self[metric_name][0].reset()
        else:
            for metric, _ in self.values():
                metric.reset()

    @staticmethod
    def __flatten_subdicts(d: Dict) -> None:
        """
        Remove sub-dictionaries, replacing them with their key-value pairs

        @param d: dictionary to flatten in-place
        """
        sub_dicts = [(k, v) for k, v in d.items() if isinstance(v, dict)]
        if sub_dicts:
            k, v = sub_dicts[0]
            d.pop(k)
            new_values = v
            for k, v in sub_dicts[1:]:
                d.pop(k)
                new_values.update(v)
            d.update(new_values)


def format_metrics(metric_set: Dict[str, torch.Tensor], precision: int = 4) -> str:
    """
    Pretty printer for metrics

    @param metric_set: torchmetrics metric dictionary
    @param precision: floating point precision to use when formatting metric values
    @return: metrics for printing in a formatted string
    """
    return " | ".join([f"{metric:s}: {value.cpu().item():.{precision}f}"
                       for metric, value in metric_set.items()])
