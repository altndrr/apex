from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Sequence
from itertools import accumulate
from typing import Literal, TypeVar

import torch
from torch.utils.data import Sampler

from src.utils.classes import BatchRequest, Request

T = TypeVar("T")

__all__ = ["RequestSampler"]


def _request_sort_key_none(request: Request) -> tuple[int, Sequence[int]]:
    """Define the key for the sorting method when no grouping is applied.

    Args:
    ----
        request (Request): The request to rank.

    """
    tokens = request.context_T
    return -len(tokens), tuple(tokens)


def _request_sort_key_context(request: Request) -> tuple[int, Sequence[int]]:
    """Define the key for the sorting method when grouping by context is applied.

    Args:
    ----
        request (Request): The request to rank.

    """
    tokens = request.context_T + request.choice_T
    return -len(tokens), tuple(tokens)


def _request_group_key(request: Request) -> Sequence[int]:
    """Define the key to group and lookup one-token completions.

    Args:
    ----
        request (Request): The request to rank.

    """
    return [id(request.image)] + request.context_T + request.choice_T[:-1]


class RequestSampler(Sampler[Request]):
    """A sampler for the `Request` dataclass.

    At initialization, it reorders the data source and groups it by a key. In case of grouped
    requests (i.e., list of list of requests), it automatically handles the flattening of the
    requests. When using the `.restore_original_order` function, the groups are re-created with
    the same dimensions. The sampler also allows to retrieve cached single-token continuations
    and their associated arguments, updating indices as necessary.

    Args:
    ----
        data_source (Sequence[Sequence[Request]] | Sequence[Request]): The data source.
        init_group_key (Callable): The group by function to initially reorganize the data source.
            Defaults to `_requests_group_key`.
        init_group_by ("context" | None): The group by value to group by requests at
            initialization. Defaults to None.
        batch_size (int): The batch size. Defaults to 0.

    """

    def __init__(
        self,
        data_source: Sequence[Sequence[Request]] | Sequence[Request],
        init_group_key: Callable = _request_group_key,
        init_group_by: Literal["context"] | None = None,
        batch_size: int = 0,
    ) -> None:
        requests_per_group = None
        if isinstance(data_source[0], Sequence):
            requests_per_group = [len(group) for group in data_source]
            data_source = sum(data_source, [])  # flatten the list

        if init_group_by is None:
            key = _request_sort_key_none
        elif init_group_by == "context":
            key = _request_sort_key_context
        else:
            raise ValueError(f"Unknown group by value: {init_group_by}")

        self._key = lambda request: key(request[1])
        self._init_group_key = lambda request: init_group_key(request[1])
        self._init_group_by = init_group_by
        self._batch_size = batch_size

        self._reorder_data_source = []
        self._data_source = data_source
        self._requests_per_group = requests_per_group

        data_source_with_indices = tuple(enumerate(data_source))
        if init_group_by is not None:
            data_source_with_indices = self._group_by(data_source_with_indices)
        self._data_source_with_indices = data_source_with_indices

    def restore_order(self, new_arr: Sequence[T]) -> list[T]:
        """Restore the original order of elements.

        Args:
        ----
            new_arr (Sequence[T]): The list to reorder.

        """
        res = [None] * len(self._data_source)
        cov = [False] * len(self._data_source)

        for ind, v in zip(self._reorder_data_source, new_arr, strict=True):
            res[ind] = v
            cov[ind] = True

        if not all(cov):
            raise ValueError("Some elements were not covered by the new list")

        return res

    def restore_order_and_groups(self, new_arr: Sequence[T]) -> list[list[T]]:
        """Restore the original order and group of elements.

        Args:
        ----
            new_arr (Sequence[T]): The list to reorder.

        """
        if self._requests_per_group is None:
            raise ValueError("Cannot regroup without the original group sizes")

        res = self.restore_order(new_arr)
        starts = [0] + list(accumulate(self._requests_per_group))
        ends = list(accumulate(self._requests_per_group))
        res = [res[start:end] for start, end in zip(starts, ends, strict=False)]

        return res

    def retrieve_cache(
        self, request: Request, logits: torch.Tensor
    ) -> Iterator[tuple[Request, Sequence[int], torch.Tensor]]:
        """Retrieve cached single-token continuations and associated arguments.

        It returns the cached single-token continuations with their associated arguments,
        additionally updating indices as necessary.

        Args:
        ----
            request (Request): The request to get the cache.
            logits (torch.Tensor): The logits of the request.

        """
        context_T, image_id, choice_T = request.context_T, id(request.image), request.choice_T
        request_id = tuple([image_id] + context_T + choice_T[:-1])
        if self._init_group_by == "context":
            assert isinstance(self._data_source_with_indices, defaultdict)
            cache_hit = self._data_source_with_indices.pop(request_id)
            if (cache_size := len(cache_hit)) == 1:
                self._reorder_data_source.extend(x[0] for x in cache_hit)
                yield request, choice_T, logits
            else:
                multi_logits = logits.expand((cache_size, -1, -1)).chunk(cache_size)
                cached_values = [(i, request, request.choice_T) for i, request in cache_hit]
                indices, cached_request, cached_choice_T = zip(*cached_values, strict=True)
                self._reorder_data_source.extend(indices)
                yield from zip(cached_request, cached_choice_T, multi_logits, strict=True)
        elif self._init_group_by is None:
            yield request, choice_T, logits
        else:
            raise ValueError(f"Unknown group by value: {self._init_group_by}")

    def __iter__(self) -> Iterator[BatchRequest]:
        """Iterate over the data source."""
        if self._init_group_by == "context":
            assert isinstance(self._data_source_with_indices, defaultdict)
            values = self._reorder([value[0] for value in self._data_source_with_indices.values()])
            batch = self._split(values)
            yield from batch
        elif self._init_group_by is None:
            values = self._reorder(self._data_source_with_indices)
            batch = self._split(values)
            yield from batch
        else:
            raise ValueError(f"Unknown group by value: {self._init_group_by}")

    def __len__(self) -> int:
        return len(self._data_source)

    def _group_by(self, _iter: Iterable[tuple[int, Request]]) -> dict:
        """Group requests by a key and a value.

        Args:
        ----
            _iter (Iterable[tuple[int, Request]]): The iterable to group.

        """
        grouped_requests = defaultdict(list)
        for request in _iter:
            if self._init_group_by == "context":
                grouped_requests[tuple(self._init_group_key(request))].append(request)
            else:
                raise ValueError(f"Unknown group by value: {self._init_group_by}")

        return grouped_requests

    def _reorder(
        self, requests: Iterable[Request] | Iterable[tuple[int, Request]]
    ) -> Iterator[Request]:
        """Reorder the requests.

        Args:
        ----
            requests (Iterable[Request] | Iterable[tuple[int, Request]]): The array to reorder.

        """
        requests = sorted(requests, key=self._key)
        if self._init_group_by != "context":
            self._reorder_data_source.extend([x[0] for x in requests])

        yield from [x[1] for x in requests]

    def _split(
        self, _iter: Iterable[Request], group_key: Callable | None = None
    ) -> Iterator[BatchRequest]:
        """Split the iterator into chunks or based on a key.

        Args:
        ----
            _iter (Iterable[Request]): The iterator to divide.
            group_key (Callable, optional): The key to split the iterable. Defaults to None.

        """
        chunked_requests = []
        _iter = tuple(_iter)
        for i, x in enumerate(_iter):
            chunked_requests.append(x)
            if len(chunked_requests) == (group_key(i, _iter) if group_key else self._batch_size):
                yield BatchRequest.from_list(chunked_requests)
                chunked_requests = []

        if chunked_requests:
            yield BatchRequest.from_list(chunked_requests)
