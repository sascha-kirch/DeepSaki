"""Set of initializers based on the [He initializer](https://arxiv.org/abs/1502.01852).

In contrast to the tensorflow implementation, an alpha value can be set to consider the non-zero slope of a
LeakyReLU activation.
"""
from typing import Any
from typing import Dict
from typing import List
from typing import NoReturn
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import tensorflow as tf


class HeAlpha(tf.keras.initializers.Initializer):
    """Parent class for HeAlpha initializers. Can not be called, must be inherited from.

    HeAlpha is a [He initializer](https://arxiv.org/abs/1502.01852) that considers the negative slope of the LeakyReLU
    activation.
    """

    def __init__(self, alpha: float = 0.3, seed: Optional[int] = None) -> None:
        """Dunder method to initialize HeAlpha object.

        Args:
            alpha (float, optional): Variable to control the width of the distribution. Should be set according the
                alpha value of the LeakyReLU activation. Defaults to 0.3.
            seed (Optional[int], optional): Seed for the random distribution. Defaults to None.
        """
        self.alpha = alpha
        self.seed = seed

    def __call__(self, shape: List[int], dtype: Optional[Union[tf.DType, np.dtype]] = None) -> NoReturn:
        """Abstract dunder method to call the object instance that must be overridden by child classes.

        Args:
            shape (List[int]): Shape of the tensor that shall be initialized.
            dtype (Optional[Union[tf.DType, np.dtype]], optional): dtype to which the data should be casted to.
                Defaults to None.

        Raises:
            NotImplementedError: if calling child class does not override `__call()__`
        """
        raise NotImplementedError()

    def get_config(self) -> Dict[str, Any]:
        """Serialize object and return dictionary containing member variables.

        Returns:
            Dict[str,Any]: {"alpha": self.alpha, "seed": self.seed}
        """
        return {"alpha": self.alpha, "seed": self.seed}

    def compute_fans(self, shape: List[int]) -> Tuple[int, int]:
        """Computes the number of input and output units for a weight shape.

        Args:
            shape (List[int]): Shape of the input tensor representing the weights of a neuronal network.

        Returns:
            Tuple[int,int]: (fan_in, fan_out)
        """
        if len(shape) < 1:  # Just to avoid errors for constants.
            fan_in = fan_out = 1
        elif len(shape) == 1:
            fan_in = fan_out = shape[0]
        elif len(shape) == 2:
            fan_in = shape[0]
            fan_out = shape[1]
        else:
            # Assuming convolution kernels (2D, 3D, or more).
            # kernel shape: (..., input_depth, depth)
            receptive_field_size = 1
            for dim in shape[:-2]:
                receptive_field_size *= dim
            fan_in = shape[-2] * receptive_field_size
            fan_out = shape[-1] * receptive_field_size
        return int(fan_in), int(fan_out)


class HeAlphaUniform(HeAlpha):
    r"""HeAlpha initializer drawing values from an uniform distribution.

    $$
    W^{[l]} \sim \mathcal{U}\left(a = -\sqrt{\frac{6}{n^{[l]}+n^{[l+1]}}}, b = \sqrt{\frac{6}{n^{[l]}+n^{[l+1]}}}\right),
    $$
    where $\alpha$ is a configurable variable and $n[l]$ is the number of parameters in layer $l$
    """

    def __init__(self, alpha: float = 0.3, seed: Optional[int] = None) -> None:
        """Dunder method to initialize HeAlphaUniform object.

        Args:
            alpha (float, optional): Variable to control the width of the distribution. Should be set according the
                alpha value of the LeakyReLU activation. Defaults to 0.3.
            seed (Optional[int], optional): Seed for the random distribution. Defaults to None.
        """
        super(HeAlphaUniform, self).__init__(alpha, seed)

    def __call__(self, shape: List[int], dtype: Optional[Union[tf.DType, np.dtype]] = None) -> tf.Tensor:
        """Dunder method to call the object instance.

        Args:
            shape (List[int]): Shape of the tensor that shall be initialized.
            dtype (Optional[Union[tf.DType, np.dtype]], optional): dtype to which the data should be casted to.
                Defaults to None.

        Returns:
            tf.Tensor: Tensor containing the weights sampled from a uniform distribution for initialization.
        """
        fan_in, _ = self.compute_fans(shape)
        limit = np.sqrt(6 / ((1 + self.alpha**2) * fan_in))
        return tf.random.uniform(shape, -limit, limit, dtype=dtype, seed=self.seed)


class HeAlphaNormal(HeAlpha):
    r"""HeAlpha initializer drawing values from an normal distribution distribution.

    $$
    W^{[l]} \sim \mathcal{N}\left(\mu = 0, \sigma^{2}= \frac{2}{(1+\alpha^{2})n[l]}\right),
    $$
    where $\mu$ is the mean, $\sigma^2$ is the variance, $\alpha$ is a configurable variable and $n[l]$ is the
    number of parameters in layer $l$
    """

    def __init__(self, alpha: float = 0.3, seed: Optional[int] = None) -> None:
        """Dunder method to initialize HeAlphaUniform object.

        Args:
            alpha (float, optional): Variable to control the width of the distribution. Should be set according the
                alpha value of the LeakyReLU activation. Defaults to 0.3.
            seed (Optional[int], optional): Seed for the random distribution. Defaults to None.
        """
        super(HeAlphaNormal, self).__init__(alpha, seed)

    def __call__(self, shape: List[int], dtype: Optional[Union[tf.DType, np.dtype]] = None) -> tf.Tensor:
        """Dunder method to call the object instance.

        Args:
            shape (List[int]): Shape of the tensor that shall be initialized.
            dtype (Optional[Union[tf.DType, np.dtype]], optional): dtype to which the data should be casted to.
                Defaults to None.

        Returns:
            tf.Tensor: Tensor containing the weights sampled from a normal distribution for initialization.
        """
        fan_in, _ = self.compute_fans(shape)
        std = np.sqrt(2) / np.sqrt((1 + self.alpha**2) * fan_in)
        return tf.random.truncated_normal(shape, mean=0, stddev=std, dtype=dtype, seed=self.seed)
