import tensorflow as tf
import numpy as np
import DeepSaki.initializer.helper

class FourierConvolution2D(tf.keras.layers.Layer):
  '''
  Performs convolution by multiplying filter and inputs in the fourier domain. Layer input is asumed to be in spatial domain.
  args:
  - filters: Number of individual filters.
  - kernels: Kernel of the spatial convolution. Expected input [height,width]. If None, kernel size is size of the input height and width
  - use_bias: bool to indicate whether or not to us bias weights. 
  - filter_initializer: Initializer for filter weights.
  - bias_initializer: Initializer for bias weights.
  - isChannelFirst: True or False. If True, input shape is assumed to be [batch,channel,height,width]. If False, input shape is assumed to be [batch,height,width,channel]
  - **kwargs: keyword arguments passed to the parent class tf.keras.layers.Layer.
    
  '''
  def __init__(self,
               filters,
               kernels = None,
               use_bias = True,
               kernel_initializer = tf.keras.initializers.RandomUniform(-0.05,0.05),
               bias_initializer = tf.keras.initializers.Zeros(),
               isChannelFirst = False,
               **kwargs
               ):
    super(FourierConvolution2D, self).__init__(**kwargs) 
    self.filters = filters
    self.kernels = kernels
    self.use_bias = use_bias
    self.kernel_initializer = kernel_initializer
    self.bias_initializer = bias_initializer
    self.isChannelFirst = isChannelFirst
    
  def build(self, input_shape):
    super(FourierConvolution2D, self).build(input_shape)
    if self.isChannelFirst:
        batch_size, inp_filter, inp_height, inp_width = input_shape
    else:
        batch_size, inp_height, inp_width, inp_filter = input_shape
    
    if not self.kernels:
        self.kernels = [inp_height, inp_width]
    
    #weights are independent from batch size [out_filter,inp_filter,kernel,kernel]. I leave the two kernels last, since I then can easily calculate the 2d FFT at once!
    self.kernel = self.add_weight(name="kernel", shape=[self.filters, inp_filter, self.kernels[0], self.kernels[1]],initializer = self.kernel_initializer, trainable=True)
    
    if self.use_bias:
        self.bias = self.add_weight(name="bias", shape=[self.filters,1,1],initializer = self.bias_initializer, trainable=True)
    
    #Output shape: batch_size, self.filters, inp_height, inp_width/2+1. Filters is zero, since concatenated later. For rFFT, the las dimension is reduced!  
    self.out_shape = (batch_size,0,inp_height, int(inp_width/2)+1)

  def call(self, inputs):
    if not self.built:
      raise ValueError('This model has not yet been built.')
    
    #FFT2D is calculated over last two dimensions! 
    if not self.isChannelFirst:
        inputs = tf.einsum("bhwc->bchw",inputs)
    
    outputs_F = np.ndarray(shape=self.out_shape)
    
    # Pad the kernel to the shape of the input to enable element-wise multiplication
    signal_shape = tf.shape(inputs)
    kernel_shape = tf.shape(self.kernel)
    x_pad = signal_shape[2] - kernel_shape[2]
    y_pad = signal_shape[3] - kernel_shape[3]
    # paddings shape is [2,4] because rank of inputs is 4, and 2 for height and width
    paddings = [[0,0],
               [0,0],
                [0,x_pad],
               [0,y_pad]
              ]
    kernels_padded = tf.pad(self.kernel, paddings) # [out_channels, inp_channel, height,width,out_channe]
    #print("Shape: inputs {}".format(np.shape(inputs)))
    #print("Shape: kernels_padded {}".format(np.shape(kernels_padded)))
    
    # Compute DFFTs for both inputs and kernel weights
    inputs_F = tf.signal.rfft2d(inputs) #[batch,height,width,channel]
    kernels_F = tf.signal.rfft2d(kernels_padded)
    #kernels_F = tf.math.conj(kernels_F) #calculate conjugate to be mathematically correct with the cross-corelation implementation. Not important, since filter is learned! 
    
    #print("Shape: inputs_F {}".format(np.shape(inputs_F)))
    #print("Shape: kernels_F {}".format(np.shape(kernels_F)))
    
    # Apply filters by element wise multiplications
    for filter in range(self.filters):
      #print("Shape: kernels_F[filter,:,:,:] {}".format(np.shape(kernels_F[filter,:,:,:])))
      outputs_F = tf.concat(
          [outputs_F,
          tf.reduce_sum(
              inputs_F  * kernels_F[filter,:,:,:], #inputs:(batch, inp_filter, height, width ), fourier_filter:(...,out_filter,inp_filter,height, width)
              axis = -3, # sum over all applied filters
              keepdims = True
          )],
          axis = -3 # is the new filter count, since channel first
      )
        
    #print("Shape: outputs_F {}".format(np.shape(outputs_F)))
    # Inverse rDFFT
    output = tf.signal.irfft2d(outputs_F)
    #output = tf.math.real(output)
    
    if self.use_bias:
        output += self.bias
        
    #reverse the channel configuration to its initial config 
    if not self.isChannelFirst:
        output = tf.einsum("bchw->bhwc",output)
    
    return output

  def get_config(self):
    config = super(FourierConvolution2D, self).get_config()
    config.update({
        "filters":self.filters,
        "kernels":self.kernels,
        "use_bias":self.use_bias,
        "kernel_initializer":self.kernel_initializer,
        "bias_initializer":self.bias_initializer,
        "isChannelFirst":self.isChannelFirst
        })
    return config
      
class FourierFilter2D(tf.keras.layers.Layer):
  '''
  Learnable filter in frequency domain. Expects input data to be in the fourier domain.
  args:
    - filters: number of independent filters
    - use_bias: bool to indicate whether or not to us bias weights. 
    - filter_initializer: Initializer for filter weights.
    - bias_initializer: Initializer for bias weights.
    - isChannelFirst: True or False. If True, input shape is assumed to be [batch,channel,height,width]. If False, input shape is assumed to be [batch,height,width,channel]
    - **kwargs: keyword arguments passed to the parent class tf.keras.layers.Layer.
  '''
  def __init__(self,
                filters, 
                use_bias = True,
                filter_initializer = tf.keras.initializers.RandomUniform(-0.05,0.05),
                bias_initializer = tf.keras.initializers.Zeros(),
                isChannelFirst = False,
               **kwargs
               ):
    super(FourierFilter2D, self).__init__(**kwargs)
    self.filters = filters 
    self.use_bias = use_bias
    self.filter_initializer = DeepSaki.initializer.helper.MakeInitializerComplex(filter_initializer)
    self.bias_initializer = DeepSaki.initializer.helper.MakeInitializerComplex(bias_initializer)
    self.isChannelFirst = isChannelFirst
    

    self.fourier_filter = None # shape: batch, height, width, input_filters, output_filters
    self.fourier_bias = None
    self.out_shape = None

  def build(self, input_shape):
    super(FourierFilter2D, self).build(input_shape)
    if self.isChannelFirst:
        batch_size,inp_filter, inp_height, inp_width = input_shape
    else:
        batch_size, inp_height, inp_width, inp_filter = input_shape
    
    #weights are independent from batch size. Filter dimensions differ from convolution, since FFT2D is calculated over last 2 dimensions
    self.fourier_filter = self.add_weight(name="filter", shape=[inp_filter, inp_height, inp_width, self.filters],initializer = self.filter_initializer, trainable=True, dtype=tf.dtypes.complex64)
    
    if self.use_bias: #shape: [filter,1,1] so it can be broadcasted when adding to the output, since FFT asumes channel first!
        self.fourier_bias = self.add_weight(name="bias", shape=[self.filters,1,1],initializer = self.bias_initializer, trainable=True, dtype=tf.dtypes.complex64) 
    
    #Output shape: batch_size, self.filters, inp_height, inp_width. Filters is zero, since concatenated later  
    self.out_shape = (batch_size,0,inp_height, inp_width)
    

  def call(self, inputs):
    '''
    I take advantage of broadcasting to calculate the batches: https://numpy.org/doc/stable/user/basics.broadcasting.html
    '''
    if not self.built:
      raise ValueError('This model has not yet been built.')
    
    if not self.isChannelFirst: #FFT2D is calculated over last two dimensions! 
        inputs = tf.einsum("bhwc->bchw",inputs)
        
    output = np.ndarray(shape=self.out_shape)
    for filter in range(self.filters):
      output = tf.concat(
          [output,
          tf.reduce_sum(
              inputs  * self.fourier_filter[:,:,:,filter], #inputs:(batch, inp_filter, height, width ), fourier_filter:(...,inp_filter,height, width, out_filter)
              axis = -3, # sum over all applied filters
              keepdims = True
          )],
          axis = -3 # is the new filter count, since channel first
      )
    
    if self.use_bias:
        output += self.fourier_bias
        
    if not self.isChannelFirst: #reverse the channel configuration to its initial config 
        output = tf.einsum("bchw->bhwc",output)

    return output

  def get_config(self):
    config = super(FourierFilter2D, self).get_config()
    config.update({
        "filters":self.filters,
        "use_bias":self.use_bias,
        "kernel_initializer":self.filter_initializer,
        "bias_initializer":self.bias_initializer,
        "isChannelFirst":self.isChannelFirst
        })
    return config
    
  
class FFT2D(tf.keras.layers.Layer):
  '''
  Calculates the 2D descrete fourier transform
  args:
  - isChannelFirst: True or False. If True, input shape is assumed to be [batch,channel,height,width]. If False, input shape is assumed to be [batch,height,width,channel]
  - applyRealFFT: True or False. If True, rfft2D is applied, which assumes real valued inputs and halves the width of the output. If False, fft2D is applied, which assumes complex input.
  - shiftFFT: True or False. If true, low frequency componentes are centered.
  - **kwargs: keyword arguments passed to the parent class tf.keras.layers.Layer.
  '''
  def __init__(self,
               isChannelFirst = False,
               applyRealFFT = False,
               shiftFFT = True,
               **kwargs
               ):
    super(FFT2D, self).__init__(**kwargs) 
    self.isChannelFirst = isChannelFirst
    self.applyRealFFT = applyRealFFT
    self.shiftFFT = shiftFFT

  def call(self, inputs):
    if not self.isChannelFirst:
        inputs = tf.einsum("bhwc->bchw",inputs)
    
    if self.applyRealFFT:
        x = tf.signal.rfft2d(inputs)
        if shiftFFT:
            x = tf.signal.fftshift(x, axes=[-2])
    else:
        imag = tf.zeros_like(inputs)
        inputs = tf.complex(inputs,imag) #fft2d requires complex inputs -> create complex with 0 imaginary
        x = tf.signal.fft2d(inputs)
        if shiftFFT:
            x = tf.signal.fftshift(x)
    
    if not self.isChannelFirst: #reverse the channel configuration to its initial config 
        x = tf.einsum("bchw->bhwc",x)
    return x

  def get_config(self):
    config = super(FFT2D, self).get_config()
    config.update({
        "isChannelFirst":self.isChannelFirst,
        "applyRealFFT":self.applyRealFFT,
        "shiftFFT":shiftFFT
        })
    return config
    
    
class iFFT2D(tf.keras.layers.Layer):
  '''
  Calculates the 2D inverse FFT and reverses the center shift operation
  args:
  - isChannelFirst: True or False. If True, input shape is assumed to be [batch,channel,height,width]. If False, input shape is assumed to be [batch,height,width,channel]
  - applyRealFFT: True or False. If True, rfft2D is applied, which assumes real valued inputs and halves the width of the output. If False, fft2D is applied, which assumes complex input.
  - shiftFFT: True or False. If True, shift operation of fourier transform is reversed before calculating the inverse fourier transformation
  - **kwargs: keyword arguments passed to the parent class tf.keras.layers.Layer.
    
  '''
  def __init__(self,
               isChannelFirst = False,
               applyRealFFT = False,
               shiftFFT = True,
               **kwargs
               ):
    super(iFFT2D, self).__init__(**kwargs) 
    self.isChannelFirst =isChannelFirst
    self.applyRealFFT = applyRealFFT
    self.shiftFFT=shiftFFT

  def call(self, inputs):
    if not self.isChannelFirst:
      inputs = tf.einsum("bhwc->bchw",inputs)
    x = inputs
    
    if self.applyRealFFT:
        if shiftFFT:
            x = tf.signal.ifftshift(x, axes=[-2])
        x = tf.signal.irfft2d(x)
    else:
        if shiftFFT:
            x = tf.signal.ifftshift(x)
        x = tf.signal.ifft2d(x)
        
    x = tf.math.real(x)
    
    if not self.isChannelFirst: #reverse the channel configuration to its initial config 
        x = tf.einsum("bchw->bhwc",x)
    return x

  def get_config(self):
    config = super(iFFT2D, self).get_config()
    config.update({
        "isChannelFirst":self.isChannelFirst,
        "applyRealFFT":self.applyRealFFT,
        "shiftFFT":shiftFFT
        })
    return config