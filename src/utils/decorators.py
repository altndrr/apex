from collections.abc import Callable
from functools import wraps

__all__ = ["rank_zero_only"]


def rank_zero_only(func: Callable, default: Callable | None = None) -> Callable | None:
    """Call function only if the rank is zero.

    Args:
    ----
        func (Callable): Function to be wrapped
        default (Callable | None): Default value to return if the rank is not zero. Defaults to
            None.

    """

    @wraps(func)
    def wrapped_fn(*args, **kwargs) -> Callable | None:
        rank = getattr(rank_zero_only, "rank", None)
        if rank is None:
            raise RuntimeError("The `rank_zero_only.rank` needs to be set before use")
        if rank == 0:
            return func(*args, **kwargs)
        return default

    return wrapped_fn
