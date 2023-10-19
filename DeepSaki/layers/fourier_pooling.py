import tensorflow as tf
    
class rFFTPooling2D(tf.keras.layers.Layer):
  '''
  Pooling in frequency domain by truncating higher frequencies. Layer input is asumed to be in spatial domain.
  args:
  - isChannelFirst: True or False. If True, input shape is assumed to be [batch,channel,height,width]. If False, input shape is assumed to be [batch,height,width,channel]
  - truncatedFrequencies: "high" or "low": if "high", high frequency values are truncated, if "low", low frequencies are truncated
  - **kwargs: keyword arguments passed to the parent class tf.keras.layers.Layer.
  '''
  def __init__(self,
               isChannelFirst = False,
               truncatedFrequencies = "low",
               **kwargs
               ):
    super(rFFTPooling2D, self).__init__(**kwargs) 
    self.isChannelFirst = isChannelFirst
    self.truncatedFrequencies=truncatedFrequencies
    
  def build(self, input_shape):
    super(rFFTPooling2D, self).build(input_shape)
    if self.isChannelFirst:
        batch_size, inp_filter, inp_height, inp_width = input_shape
    else:
        batch_size, inp_height, inp_width, inp_filter = input_shape
    self.offset_height = int(inp_height/2)
    self.offset_width = 0
    self.target_height = int(inp_height/2)
    self.target_width = int(inp_width/4 + 1) #1/4 because real spectrum has allready half width and filter only applies to positive frequencies in width

  def call(self, inputs):
    if not self.built:
      raise ValueError('This model has not yet been built.')
    
    if not self.isChannelFirst: #layer assumes channel first due to FFT
        inputs = tf.einsum("bhwc->bchw",inputs)
    
    inputs_F = tf.signal.rfft2d(inputs)
    if self.truncatedFrequencies == "high":
        inputs_F = tf.signal.fftshift(inputs_F, axes=[-2]) #shift frequencies to be able to crop in center
    shape = tf.shape(inputs_F)
    outputs_F = tf.slice(inputs_F, begin=[0,0,self.offset_height,self.offset_width],size=[shape[0],shape[1],self.target_height,self.target_width]) # Tf.slice instead of tf.image.crop, because the latter assumes channel last
    if self.truncatedFrequencies == "high":
        outputs_F = tf.signal.ifftshift(outputs_F, axes=[-2]) #reverse shift
    outputs = tf.signal.irfft2d(outputs_F) 
    
    #reverse to previous channel config!
    if not self.isChannelFirst:
        outputs = tf.einsum("bchw->bhwc",outputs)
    return outputs

  def get_config(self):
    config = super(rFFTPooling2D, self).get_config()
    config.update({
        "isChannelFirst":self.isChannelFirst,
        "truncatedFrequencies":self.truncatedFrequencies
        })
    return config
    

class FourierPooling2D(tf.keras.layers.Layer):
  '''
  Pooling in frequency domain by truncating high frequencies. Layer input is asumed to be in frequency domain.
  args:
  - isChannelFirst: True or False. If True, input shape is assumed to be [batch,channel,height,width]. If False, input shape is assumed to be [batch,height,width,channel]
  - **kwargs: keyword arguments passed to the parent class tf.keras.layers.Layer.
  '''
  def __init__(self,
               isChannelFirst = False,
               **kwargs
               ):
    super(FourierPooling2D, self).__init__(**kwargs) 
    self.isChannelFirst = isChannelFirst

  def call(self, inputs):
    if self.isChannelFirst:
        inputs = tf.einsum("bchw->bhwc",inputs)
    
    outputs = tf.image.central_crop(inputs, 0.5) #assumes channel last
    
    #reverse to previous channel config!
    if self.isChannelFirst:
        outputs = tf.einsum("bhwc->bchw",outputs)
    return outputs

  def get_config(self):
    config = super(FourierPooling2D, self).get_config()
    config.update({
        "isChannelFirst":self.isChannelFirst
        })
    return config
    
 
    
    
