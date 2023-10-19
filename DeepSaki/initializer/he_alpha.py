import tensorflow as tf
import numpy as np

class HeAlpha(tf.keras.initializers.Initializer):
  '''
    Parent class for HeAlpha initializers. Can not be called, must be inherited from.
    HeAlpha is an He initializer that considers the negative slope of the LeakyReLU activation
    args:
      - alpha: Variable to control the width of the distribution. Should be set acc. the alpha value of the LeakyReLU activation.
      - seed: seed for the random distribution
  '''
  def __init__(self,alpha = 0.3, seed = None):
    self.alpha = alpha
    self.seed = seed    

  def __call__(self, shape, dtype=None, **kwargs):
    raise NotImplementedError()

  def get_config(self):  # To support serialization
    return {"alpha": self.alpha, "seed":self.seed}

  def compute_fans(self, shape):
    """Computes the number of input and output units for a weight shape.
    Args:
      shape: Integer shape tuple or TF tensor shape.
    Returns:
      A tuple of integer scalars (fan_in, fan_out).
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
  '''
  HeAlpha initializer drawing values from an uniform distribution.
  '''
  def __init__(self,alpha = 0.3, seed = None):
    super(HeAlphaUniform, self).__init__(alpha, seed)  

  def __call__(self, shape, dtype=None, **kwargs):
    fan_in, _ = self.compute_fans(shape)
    limit = np.sqrt(6/((1+self.alpha**2)*fan_in))
    return tf.random.uniform( shape,-limit,limit, dtype=dtype,seed = self.seed)

  
  
class HeAlphaNormal(HeAlpha):
  '''
  HeAlpha initializer drawing values from an normal distribution distribution.
  '''
  def __init__(self, alpha = 0.3, seed = None):
    super(HeAlphaNormal, self).__init__(alpha, seed)  

  def __call__(self, shape, dtype=None, **kwargs):
    fan_in, _ = self.compute_fans(shape)
    std = np.sqrt(2)/np.sqrt((1+self.alpha**2)*fan_in)
    return tf.random.truncated_normal(shape, mean=0, stddev=std, dtype=dtype,seed = self.seed)
