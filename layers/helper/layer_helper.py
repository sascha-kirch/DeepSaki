import tensorflow as tf
import DeepSaki.layers
import DeepSaki.initializer

def GetInitializer(initializerString, seed = None):
  '''
  Wrapper to return a certain initializer given a descriptive string
  args:
    - initializerString: string to describe desired initialier
    - seed (optional, default: None): seed that can be fed to the initializer 
  '''
  if initializerString == "random_normal":
    return tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02, seed=seed)
  elif initializerString == "random_uniform":
    return tf.keras.initializers.RandomUniform(minval=-0.002, maxval=0.02, seed=seed)
  elif initializerString == "glorot_normal":
    return tf.keras.initializers.GlorotNormal(seed=seed)
  elif initializerString == "glorot_uniform":
    return tf.keras.initializers.GlorotUniform(seed=seed)
  elif initializerString == "he_normal":
    return tf.keras.initializers.HeNormal(seed=seed)
  elif initializerString == "he_uniform":
    return tf.keras.initializers.HeUniform(seed=seed)
  elif initializerString == "he_alpha_normal":
    return DeepSaki.initializer.HeAlphaNormal(seed=seed)
  elif initializerString == "he_alpha_uniform":
    return DeepSaki.initializer.HeAlphaUniform(seed=seed)
  else:
    raise Exception("Undefined initializerString provided: {}".format(initializerString))
    
    
def pad_func(padValues=(1, 1), padding = "zero"):
  '''
  Wrapper to obtain a padding layer by string
  args:
    - padValues (optional, default: (1, 1)): size of the padding
    - padding (optional, default: "zero"): string
  '''
  if padding == "reflection":
    return DeepSaki.layers.ReflectionPadding2D(padValues)
  elif padding == "zero":
    return tf.keras.layers.ZeroPadding2D(padValues)
  else:
    raise Exception("Undefined padding type provided: {}".format(padding))
    
def dropout_func(filters, dropout_rate):
  '''
  Wrapper to obtain a dropout layer depending on the size of the preceeding feature map
  args:
    - filters: number of filters of previous layer
    - dropout_rate: probability with which dropout is performed
  '''
  if filters > 1:
    return tf.keras.layers.SpatialDropout2D(dropout_rate)
  else:
    return tf.keras.layers.Dropout(dropout_rate)
  
def PlotLayer(layer, inputShape):
  '''
  Creates an model from a given layer to be able to call model.summary() and to plot a graphic
  args:
    layer: tf.keras.layer object to be ploted
    inputShape: shape of the input data without batchsize -> (height, width, channel)
  '''
  layer.build([None,*inputShape])
  inputs = tf.keras.layers.Input(shape=inputShape)
  model = tf.keras.Model(inputs=inputs, outputs=layer.call(inputs))
  model.summary()
  tf.keras.utils.plot_model(model, show_shapes=True, expand_nested=True,to_file=layer.name + ".png")
