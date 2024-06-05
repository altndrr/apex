import re
from collections import ChainMap
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from string import Template, _sentinel_dict
from typing import Any

import torch

from src.utils._torch_utils import pad_and_concat
from src.utils.types import PILImage

__all__ = ["Prompt", "BatchRequest", "Request", "Response"]


class Prompt(Template):
    """A string class for prompts, supporting $-substitutions.

    It extends the `string.Template` class to support mandatory keys. It also
    add a `.strip()` method to remove leading/trailing whitespace (due to missing
    optional keys).

    Args:
    ----
        template (str): The string template.
        mandatory (Sequence[str]): A list of mandatory keys. Defaults to [].

    """

    def __init__(self, template: str, mandatory: Sequence[str] = []) -> None:
        super().__init__(template)
        self.mandatory = mandatory

    def substitute(self, mapping: Mapping = _sentinel_dict, /, **kwargs) -> str:
        """Substitute the template with the given mapping.

        Args:
        ----
            mapping (dict, optional): The mapping. Defaults to _sentinel_dict.
            kwargs: The substitution values.

        """
        if mapping is _sentinel_dict:
            mapping = kwargs
        elif kwargs:
            mapping = ChainMap(kwargs, mapping)

        def convert(mo: re.Match) -> str:
            """Convert the match object to the corresponding value.

            Args:
            ----
                mo (str): The match object.

            """
            named = mo.group("named") or mo.group("braced")
            if named is not None:
                if named in mapping:
                    return str(mapping[named])
                if named in self.mandatory:
                    raise ValueError(f"Missing mandatory key: {named}")
                return ""
            if mo.group("escaped") is not None:
                return self.delimiter
            if mo.group("invalid") is not None:
                self._invalid(mo)
            raise ValueError("Unrecognized named group in pattern", self.pattern)

        out = self.pattern.sub(convert, self.template)
        out = out.strip()  # remove leading/trailing whitespace due to missing optional keys

        return out

    def __repr__(self) -> str:
        return self.safe_substitute()

    def _invalid(self, mo: re.Match) -> None:
        """Raise an error for invalid placeholders.

        Args:
        ----
            mo (str): The match object.

        """
        i = mo.start("invalid")
        lines = self.template[:i].splitlines(keepends=True)
        if not lines:
            colno = 1
            lineno = 1
        else:
            colno = i - len("".join(lines[:-1]))
            lineno = len(lines)
        raise ValueError(f"Invalid placeholder in string: line {lineno}, col {colno}")


@dataclass
class Request:
    """A dataclass for requests to LLMs with the completion choice.

    Args:
    ----
        context (str): The string defining the request.
        choice (str | None): Optional model completion string.
        image (str | PILImage | None): Optional image to include in the request.
        context_T (Sequence[int] | None): The encoding of the context.
        choice_T (Sequence[int] | None): The encoding of the choice.
        image_CHW (torch.Tensor | None): The encoding of the image.

    Dimension keys:
    ----------
        T: text sequence length
        C: image channels
        H: image height
        W: image width

    """

    context: str
    choice: str | None = None
    image: str | PILImage | None = None
    context_T: Sequence[int] | None = None
    choice_T: Sequence[int] | None = None
    image_CHW: torch.Tensor | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # remove leading/trailing whitespace
        self.context = self.context.strip()

        # remove batch dimension if present
        if self.image_CHW is not None and len(self.image_CHW.shape) == 4:
            self.image_CHW = self.image_CHW.squeeze(0)

    @property
    def whole_T(self) -> Sequence[int] | None:
        """Get the whole encoding of the request."""
        if self.choice_T is not None:
            return self.context_T + self.choice_T

        return self.context_T

    def __iter__(self) -> Iterator:
        """Iterate over the items of the request."""
        for item in self.__dataclass_fields__:
            yield getattr(self, item)


@dataclass
class BatchRequest:
    """A dataclass for batched requests to LLMs with the completion choice.

    Args:
    ----
        context (list[str]): The list of strings defining the requests.
        choice (list[str] | None): Optional list of model completion strings.
        image (list[str] | list[PILImage] | None): Optional list of images to include in the
            requests.
        context_T (list[Sequence[int]] | None): The list of encodings of the context.
        choice_T (list[Sequence[int]] | None): The list of encodings of the choice.
        image_CHW (torch.Tensor | None): The encoding of the image.

    Dimension keys:
    ----------
        T: text sequence length
        C: image channels
        H: image height
        W: image width

    """

    context: list[str]
    choice: list[str] | None
    image: list[str] | list[PILImage] | None = None
    context_BT: list[Sequence[int]] | None = None
    choice_BT: list[Sequence[int]] | None = None
    image_BCHW: torch.Tensor | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)
    _requests: list[Request] = field(default_factory=list)

    def __post_init__(self) -> None:
        # remove leading/trailing whitespace
        self.context = [c.strip() for c in self.context]

        # replace list of None with None if all elements are None
        if all(map(lambda x: x is None, self.choice)):
            self.choice = None
            self.choice_BT = None

    @property
    def whole_BT(self) -> list[Sequence[int]]:
        """Get the whole encodings of the requests."""
        if self.choice_BT is not None:
            return [c + ch for c, ch in zip(self.context_BT, self.choice_BT, strict=False)]

        return self.context_BT

    @staticmethod
    def from_list(requests: list[Request]) -> "BatchRequest":
        """Convert a list of requests to a batched request.

        Args:
        ----
            requests (list[Request]): The list of requests.

        """
        # pad and concat the extra kwargs
        image_BCHW = torch.stack([r.image_CHW for r in requests], dim=0)
        kwargs = {k: [torch.tensor(r.kwargs[k]) for r in requests] for k in requests[0].kwargs}
        kwargs = {k: pad_and_concat(v, padding="right") for k, v in kwargs.items()}

        return BatchRequest(
            context=[r.context for r in requests],
            choice=[r.choice for r in requests],
            image=[r.image for r in requests],
            context_BT=[r.context_T for r in requests],
            choice_BT=[r.choice_T for r in requests],
            image_BCHW=image_BCHW,
            kwargs=kwargs,
            _requests=requests,
        )

    def __iter__(self) -> Iterator:
        """Iterate over the requests."""
        yield from self._requests

    def __len__(self) -> int:
        """Return the number of requests."""
        return len(self._requests)


@dataclass
class Response:
    """A dataclass for responses from LLMs.

    Args:
    ----
        choice (str): The model completion string.
        is_equal_to_greedy (bool | torch.Tensor): Whether the choice is equal to the greedy
            completion.
        log_prob_sum (float | torch.Tensor): The sum of log probabilities of the tokens in the
            choice.
        log_prob_mean (float | torch.Tensor): The mean of log probabilities of the tokens in the
            choice.

    """

    text: str
    is_equal_to_greedy: bool | torch.Tensor
    log_prob_sum: float | torch.Tensor | None = None
    log_prob_mean: float | torch.Tensor | None = None

    def __post_init__(self) -> None:
        # remove leading/trailing whitespace
        self.text = self.text.strip()

        # convert torch.Tensor to Python scalar
        if not isinstance(self.is_equal_to_greedy, bool):
            self.is_equal_to_greedy = self.is_equal_to_greedy.item()
        if self.log_prob_sum is not None and not isinstance(self.log_prob_sum, float):
            self.log_prob_sum = self.log_prob_sum.item()
        if self.log_prob_mean is not None and not isinstance(self.log_prob_mean, float):
            self.log_prob_mean = self.log_prob_mean.item()

    def __iter__(self) -> Iterator:
        """Iterate over the items of the request."""
        for item in self.__dataclass_fields__:
            yield getattr(self, item)
