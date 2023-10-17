import tensorflow as tf

import DeepSaki.initializers
import DeepSaki.layers

def get_initializer(initializer_string, seed=None):
    """
    Wrapper to return a certain initializer given a descriptive string
    args:
      - initializer_string: string to describe desired initialier
      - seed (optional, default: None): seed that can be fed to the initializer
    """
    if initializer_string == "random_normal":
        return tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02, seed=seed)
    elif initializer_string == "random_uniform":
        return tf.keras.initializers.RandomUniform(minval=-0.002, maxval=0.02, seed=seed)
    elif initializer_string == "glorot_normal":
        return tf.keras.initializers.GlorotNormal(seed=seed)
    elif initializer_string == "glorot_uniform":
        return tf.keras.initializers.GlorotUniform(seed=seed)
    elif initializer_string == "he_normal":
        return tf.keras.initializers.HeNormal(seed=seed)
    elif initializer_string == "he_uniform":
        return tf.keras.initializers.HeUniform(seed=seed)
    elif initializer_string == "he_alpha_normal":
        return DeepSaki.initializers.HeAlphaNormal(seed=seed)
    elif initializer_string == "he_alpha_uniform":
        return DeepSaki.initializers.HeAlphaUniform(seed=seed)
    else:
        raise Exception("Undefined initializer_string provided: {}".format(initializer_string))


def pad_func(pad_values=(1, 1), padding="zero"):
    """
    Wrapper to obtain a padding layer by string
    args:
      - pad_values (optional, default: (1, 1)): size of the padding
      - padding (optional, default: "zero"): string
    """
    if padding == "reflection":
        return DeepSaki.layers.ReflectionPadding2D(pad_values)
    elif padding == "zero":
        return tf.keras.layers.ZeroPadding2D(pad_values)
    else:
        raise Exception("Undefined padding type provided: {}".format(padding))


def dropout_func(filters, dropout_rate):
    """
    Wrapper to obtain a dropout layer depending on the size of the preceeding feature map
    args:
      - filters: number of filters of previous layer
      - dropout_rate: probability with which dropout is performed
    """
    if filters > 1:
        return tf.keras.layers.SpatialDropout2D(dropout_rate)
    else:
        return tf.keras.layers.Dropout(dropout_rate)


def plot_layer(layer, input_shape):
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
