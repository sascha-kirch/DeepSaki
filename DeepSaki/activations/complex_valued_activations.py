import tensorflow as tf

class ComplexActivation(tf.keras.layers.Layer):
  '''
  Wrapper to apply activations to complex values individually for the real and imaginary part
  args: 
  - activation: activation function to complexify
  - **kwargs: keyword arguments passed to the parent class tf.keras.layers.Layer.
  '''
  def __init__(self,
               activation = tf.keras.layers.ReLU(),
               **kwargs
               ):
    super(ComplexActivation, self).__init__(**kwargs) 
    self.activation = activation

  def call(self, inputs):
    real = self.activation(tf.math.real(inputs))
    imag = self.activation(tf.math.imag(inputs))
    return tf.complex(real,imag)

  def get_config(self):
    config = super(ComplexActivation, self).get_config()
    config.update({
        "activation":self.activation
        })
    return config