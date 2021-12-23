import tensorflow as tf

class GlobalSumPooling2D(tf.keras.layers.Layer):
  def __init__(self, data_format=None, **kwargs):
    super(GlobalSumPooling2D, self).__init__(**kwargs)
    self.data_format = 'channels_last'
    self.input_spec = tf.keras.layers.InputSpec(ndim=4)

  def compute_output_shape(self, input_shape):
    input_shape = tf.TensorShape(input_shape).as_list()
    if self.data_format == 'channels_last':
      return tf.TensorShape([input_shape[0], input_shape[3]])

  def call(self, inputs):
    return tf.reduce_sum(input_tensor=inputs, axis=[1,2], keepdims=False)

  def get_config(self):
    config = {'data_format': self.data_format}
    return config
