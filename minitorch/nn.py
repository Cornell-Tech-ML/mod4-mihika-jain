from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off

max_reduce = FastOps.reduce(operators.max, float("-inf"))


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0

    # Calculate new dimensions
    new_height = height // kh
    new_width = width // kw

    # Reshape and permute to group pooling regions
    # Use view to reshape without permute
    tiled = input.contiguous().view(batch, channel, new_height, kh, new_width, kw)
    tiled = tiled.permute(0, 1, 2, 4, 3, 5).contiguous()
    tiled = tiled.view(batch, channel, new_height, new_width, kh * kw)
    return tiled, new_height, new_width


def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Perform 2D average pooling on the input tensor with a given kernel size.

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width

    """
    batch, channel = input.shape[:2]
    tiled, new_height, new_width = tile(input, kernel)
    pooled = tiled.mean(dim=4)
    return pooled.view(batch, channel, new_height, new_width)


def argmax(input: Tensor, dim: int) -> Tensor:
    """Compute the argmax as a 1-hot tensor along a specified dimension.

    Args:
    ----
        input (Tensor): The input tensor.
        dim (int): The dimension to compute the argmax.

    Returns:
    -------
        Tensor: A 1-hot tensor with the same shape as the input, where the maximum values along the specified dimension are 1 and all other values are 0.

    """
    max_values = max_reduce(input, dim)
    one_hot = input == max_values

    return one_hot


class Max(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: int) -> Tensor:
        """Apply max reduction along a specified dimension.

        Args:
        ----
            input (Tensor): The input tensor.
            ctx (Context): The context object to save values for the backward pass.
            dim (int): The dimension to reduce.

        Returns:
        -------
            Tensor: The tensor with the maximum values along the specified dimension.

        """
        if isinstance(dim, Tensor):
            dimI = int(dim._tensor._storage[0])
        else:
            dimI = int(dim)

        ctx.save_for_backward(input, input._ensure_tensor(dimI))
        return max_reduce(input, dimI)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute the gradient of the max reduction operation.

        Args:
        ----
            ctx (Context): The context object containing saved values from the forward pass.
            grad_output (Tensor): The gradient of the loss with respect to the output of the max reduction.

        Returns:
        -------
            Tuple[Tensor, None]: The gradient of the loss with respect to the input tensor and None for the dimension.

        """
        input, dim = ctx.saved_values
        dim_val = int(dim.item())

        # Compute the gradient
        grad_input = (argmax(input, dim_val) * grad_output).sum(dim=dim_val)
        return grad_input, input._ensure_tensor(0.0)


def max(input: Tensor, dim: int) -> Tensor:
    """Apply max reduction along a specified dimension.

    Args:
    ----
        input (Tensor): The input tensor.
        dim (int): The dimension to reduce.

    Returns:
    -------
        Tensor: The tensor with the maximum values along the specified dimension.

    """
    return Max.apply(input, input._ensure_tensor(dim))


def softmax(input: Tensor, dim: int) -> Tensor:
    """Compute the softmax of the input tensor along a specified dimension.

    Args:
    ----
        input (Tensor): The input tensor.
        dim (int): The dimension to apply softmax.

    Returns:
    -------
        Tensor: The tensor with softmax applied along the specified dimension.

    """
    inp_exps = input.exp()
    sum_exps = inp_exps.sum(dim)
    shape = list(inp_exps.shape)
    shape[dim] = 1
    return inp_exps / sum_exps.contiguous().view(*shape)


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    """Compute the log of the softmax of the input tensor along a specified dimension.

    Args:
    ----
        input (Tensor): The input tensor.
        dim (int): The dimension to apply logsoftmax.

    Returns:
    -------
        Tensor: The tensor with logsoftmax applied along the specified dimension.

    """
    return softmax(input, dim).log()


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Perform 2D max pooling on the input tensor with a given kernel size.

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width

    """
    batch, channel = input.shape[:2]
    tiled, new_height, new_width = tile(input, kernel)
    pooled = max(tiled, 4)
    return pooled.view(batch, channel, new_height, new_width)


def dropout(input: Tensor, p: float, ignore: bool = False) -> Tensor:
    """Apply dropout to the input tensor based on random noise.

    Args:
    ----
        input (Tensor): The input tensor.
        p (float): The dropout rate (probability of dropping each element).
        ignore (bool): If True, don't apply dropout.

    Returns:
    -------
        Tensor: The tensor with random positions dropped.

    """
    if not ignore and p > 0.0:
        noise = rand(input.shape) > p
        return input * noise
    else:
        return input
