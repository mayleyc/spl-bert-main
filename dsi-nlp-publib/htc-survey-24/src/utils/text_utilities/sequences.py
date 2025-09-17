from typing import Iterable


def batch_list(iterable: Iterable, batch_size: int = 10) -> Iterable:
    """
    Yields batches from an iterable container

    :param iterable: elements to be batched
    :param batch_size: (max) number of elements in a single batch
    :return: generator of batches
    """
    data_len = len(iterable)
    for ndx in range(0, data_len, batch_size):
        yield iterable[ndx:min(ndx + batch_size, data_len)]
