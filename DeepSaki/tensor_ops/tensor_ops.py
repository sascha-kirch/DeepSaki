from typing import Iterable
from typing import Tuple
from typing import Union

import tensorflow as tf

def sample_array_to_tensor(
    array: tf.Tensor,
    index: Union[tf.Tensor, Iterable[int], int],
    shape: Tuple[int, int, int, int] = (-1, 1, 1, 1),
    dtype: tf.DType = tf.float32,
) -> tf.Tensor:
    """Takes a sample from a given `array` at a given `index` and creates a tensor of shape `shape`.

    The shape (-1,1,1,1) creates a tensor of said shape, which is then prodcasted to the shape of the operand

    Args:
        array (tf.Tensor): Array from which sample is taken.
        index (Union[tf.Tensor, Iterable[int], int]): Index at which sample is taken from `array`.
        shape (Tuple[int, int, int, int], optional): Shape of the resulting tensor. Defaults to (-1, 1, 1, 1).

    Returns:
        tf.Tensor: New tensor of shape `shape`
    """
    # for whatever reason when I reshape, the tensor is casted to float64 and then the multiplication bellow throws
    # an exception. Therefore I cast it to float32.
    return tf.cast(tf.reshape(tf.experimental.numpy.take(array, index), shape), dtype=dtype)
