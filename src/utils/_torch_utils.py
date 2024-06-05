from typing import Literal

import torch

__all__ = ["pad_and_concat"]


def pad_and_concat(
    tensors: list[torch.Tensor],
    max_len: int = -1,
    padding: Literal["left", "right"] = "right",
) -> torch.Tensor:
    """Pad and concatenate tensors.

    Given a list of tensors, pad them to the same length and concatenate them along the first
    dimension. The padding can be done on the left or right side. If not specified, the max length
    is the maximum length of the tensors. The function assumes that the tensors have only one
    non-1 dimension (e.g., shape (D), or (D, 1)).

    Args:
    ----
        tensors (list[torch.Tensor]): The tensors to pad and concatenate.
        max_len (int): The maximum length. If -1, use the maximum length of the tensors.
            Defaults to -1.
        padding ("left" | "right"): The padding. Defaults to "right".

    """
    if max_len == -1:
        max_len = max(tensor.shape[0] for tensor in tensors)

    out = []
    for tensor in tensors:
        tensor_len, extra_len = tensor.shape[0], tensor.shape[1:]

        tensor = tensor.squeeze()  # remove all the dimensions with size 1
        if tensor.dim() > 1:
            raise ValueError("Only tensors with one non-1 dimension are supported.")

        if tensor_len < max_len:
            pad = torch.zeros(max_len - tensor_len, dtype=tensor.dtype, device=tensor.device)
            if padding == "left":
                tensor = torch.cat([pad, tensor], dim=0)
            elif padding == "right":
                tensor = torch.cat([tensor, pad], dim=0)
            else:
                raise ValueError(f"Invalid padding {padding}")

        out.append(tensor.view(max_len, *extra_len).unsqueeze(0))

    return torch.cat(out, dim=0)
