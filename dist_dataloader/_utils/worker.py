r""""Contains definitions of the methods used by the _BaseDataLoaderIter workers.

These **needs** to be in global scope since Py2 doesn't support serializing
static methods.
"""

import torch
import random
import queue
from torch._utils import ExceptionWrapper
from torch.utils.data import _DatasetKind
from torch.utils.data._utils.worker import _IterableDatasetStopIteration, _ResumeIteration
from typing import Union
import ray
from torch.utils.data._utils import MP_STATUS_CHECK_INTERVAL


@ray.remote
def _worker_loop(dataset_kind, dataset, index_queue, data_queue,
                 auto_collation, collate_fn, drop_last, seed, init_fn, worker_id):

    torch.set_num_threads(1)
    random.seed(seed)
    torch.manual_seed(seed)
    init_exception = None

    try:
        if init_fn is not None:
            init_fn(worker_id)

        fetcher = _DatasetKind.create_fetcher(dataset_kind, dataset, auto_collation, collate_fn, drop_last)
    except Exception:
        init_exception = ExceptionWrapper(
            where="in DataLoader worker process {}".format(worker_id))

    # When using Iterable mode, some worker can exit earlier than others due
    # to the IterableDataset behaving differently for different workers.
    # When such things happen, an `_IterableDatasetStopIteration` object is
    # sent over to the main process with the ID of this worker, so that the
    # main process won't send more tasks to this worker, and will send
    # `None` to this worker to properly exit it.
    #
    # Note that we cannot set `done_event` from a worker as it is shared
    # among all processes. Instead, we set the `iteration_end` flag to
    # signify that the iterator is exhausted. When either `done_event` or
    # `iteration_end` is set, we skip all processing step and just wait for
    # `None`.
    iteration_end = False
    while True:
        try:
            r = index_queue.get(timeout=MP_STATUS_CHECK_INTERVAL)
        except queue.Empty:
            continue
        if isinstance(r, _ResumeIteration):
            # Acknowledge the main process
            data_queue.put((r, None))
            iteration_end = False
            # Recreate the fetcher for worker-reuse policy
            fetcher = _DatasetKind.create_fetcher(
                dataset_kind, dataset, auto_collation, collate_fn, drop_last)
            continue
        elif r is None:
            # Received the final signal
            assert iteration_end
            break
        elif iteration_end:
            # `done_event` is set. But I haven't received the final signal
            # (None) yet. I will keep continuing until get it, and skip the
            # processing steps.
            continue
        idx, index = r
        data: Union[_IterableDatasetStopIteration, ExceptionWrapper]
        if init_exception is not None:
            data = init_exception
            init_exception = None
        else:
            try:
                data = fetcher.fetch(index)
            except Exception as e:
                if isinstance(e, StopIteration) and dataset_kind == _DatasetKind.Iterable:
                    data = _IterableDatasetStopIteration(worker_id)
                    # Set `iteration_end`
                    #   (1) to save future `next(...)` calls, and
                    #   (2) to avoid sending multiple `_IterableDatasetStopIteration`s.
                    iteration_end = True
                else:
                    # It is important that we don't store exc_info in a variable.
                    # `ExceptionWrapper` does the correct thing.
                    # See NOTE [ Python Traceback Reference Cycle Problem ]
                    data = ExceptionWrapper(
                        where="in DataLoader worker process {}".format(worker_id))
        data_queue.put((idx, data))
        del data, idx, index, r  # save memory

