"""Collection of functions to simplify the code in various layers."""
from enum import Enum
from typing import List
from typing import Optional
from typing import Tuple

import tensorflow as tf

from DeepSaki.initializers.he_alpha import HeAlphaNormal
from DeepSaki.initializers.he_alpha import HeAlphaUniform
from DeepSaki.layers.padding import ReflectionPadding2D

class PaddingType(Enum):
    """`Enum` used to define different types of padding opperations.

    Attributes:
        ZERO (int): Indicates to apply a zero padding operations.
        REFLECTION (int): Indicates to apply a reflection padding operation.
    """

    NONE = 0
    ZERO = 1
    REFLECTION = 2


class InitializerFunc(Enum):
    """`Enum` used to define different types of initializer functions.

    Attributes:
        RANDOM_NORMAL (int): Corresponds to a random normal initializer function.
        RANDOM_UNIFORM (int): Corresponds to a random uniform initializer function.
        GLOROT_NORMAL (int): Corresponds to a Glorot normal initializer function.
        GLOROT_UNIFORM (int): Corresponds to a Glorot uniform initializer function.
        HE_NORMAL (int): Corresponds to a He normal initializer function.
        HE_UNIFORM (int): Corresponds to a He uniform initializer function.
        HE_ALPHA_NORMAL (int): Corresponds to a He Alpha normal initializer function.
        HE_ALPHA_UNIFORM (int): Corresponds to a He Alpha Uniform initializer function.
    """

    NONE = 0
    RANDOM_NORMAL = 1
    RANDOM_UNIFORM = 2
    GLOROT_NORMAL = 3
    GLOROT_UNIFORM = 4
    HE_NORMAL = 5
    HE_UNIFORM = 6
    HE_ALPHA_NORMAL = 7
    HE_ALPHA_UNIFORM = 8


def get_initializer(initializer: InitializerFunc, seed: Optional[int] = None) -> tf.keras.initializers.Initializer:
    """Wrapper to return a certain initializer given a descriptive string.

    Args:
        initializer (InitializerFunc): Enum description of the initializer.
        seed (Optional[int], optional): Seed to make the behavior of the initializer deterministic. Defaults to None.

    Returns:
        Instance of an initializer object.
    """
    valid_options = {
        InitializerFunc.RANDOM_NORMAL: tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02, seed=seed),
        InitializerFunc.RANDOM_UNIFORM: tf.keras.initializers.RandomUniform(minval=-0.002, maxval=0.02, seed=seed),
        InitializerFunc.GLOROT_NORMAL: tf.keras.initializers.GlorotNormal(seed=seed),
        InitializerFunc.GLOROT_UNIFORM: tf.keras.initializers.GlorotUniform(seed=seed),
        InitializerFunc.HE_NORMAL: tf.keras.initializers.HeNormal(seed=seed),
        InitializerFunc.HE_UNIFORM: tf.keras.initializers.HeUniform(seed=seed),
        InitializerFunc.HE_ALPHA_NORMAL: HeAlphaNormal(seed=seed),
        InitializerFunc.HE_ALPHA_UNIFORM: HeAlphaUniform(seed=seed),
    }

    assert initializer in valid_options, f"Undefined initializer provided: {initializer}"

    return valid_options.get(initializer)


def pad_func(
    pad_values: Tuple[int, int] = (1, 1), padding_type: PaddingType = PaddingType.ZERO
) -> tf.keras.layers.Layer:
    """Wrapper to obtain a padding layer instance.

    Args:
        pad_values (Tuple[int,int], optional): Size of the padding values. Defaults to (1, 1).
        padding_type (PaddingType, optional): [_description_]. Defaults to PaddingType.ZERO.

    Returns:
        Instance of a padding layer object.
    """
    valid_options = {
        PaddingType.REFLECTION: ReflectionPadding2D(pad_values),
        PaddingType.ZERO: tf.keras.layers.ZeroPadding2D(pad_values),
    }
    assert (
        padding_type in valid_options
    ), f"Undefined padding type provided: '{padding_type}'. Valid options are: '{valid_options.keys()}'"
    return valid_options.get(padding_type)


def dropout_func(filters: int, dropout_rate: float) -> tf.keras.layers.Layer:
    """
    Wrapper to obtain a dropout layer depending on the size of the preceeding feature map
    args:
      - filters: number of filters of previous layer
      - dropout_rate: probability with which dropout is performed
    """
    if filters > 1:
        return tf.keras.layers.SpatialDropout2D(dropout_rate)
    return tf.keras.layers.Dropout(dropout_rate)


def plot_layer(layer: tf.keras.layers.Layer, input_shape: List[int]) -> None:
    """
    Creates an model from a given layer to be able to call model.summary() and to plot a graphic
    args:
      layer: tf.keras.layer object to be ploted
      input_shape: shape of the input data without batchsize -> (height, width, channel)
    """
    layer.build([None, *input_shape])
    inputs = tf.keras.layers.Input(shape=input_shape)
    model = tf.keras.Model(inputs=inputs, outputs=layer.call(inputs))
    model.summary()
    tf.keras.utils.plot_model(model, show_shapes=True, expand_nested=True, to_file=layer.name + ".png")