import tensorflow as tf
import numpy as np

class PixelDistanceLoss(tf.keras.losses.Loss):
  '''
  calculates the pixel distance loss (per pixel loss) of two images of the shape (batch, height, width, channels)
  args:
    globalBatchSize: Batchsize of the training data to manually reduce the loss. When following a distributed strategy globalBatchSize = Num_worker * batchSizePerWorker
    calculationType (optional, default:"per_image"): determines whether the loss calculated per image or summed over all channels. Options: "per_image", "per_channel". 
    normalizeDepthChannel (optional, default: False): For RGBD images, the weight of depth is increased by multiplying the depth by the number of color channels.
    MAE_MSE (optional, default:"mae"): determines whether use MAE (mean absolut error) or MSE (mean squared error) for the pixel distance. Options: "mae", "mse". 
  '''

  def __init__(self,globalBatchSize, calculationType = "per_image", normalizeDepthChannel = False, MAE_MSE = "mae", reduction = tf.keras.losses.Reduction.AUTO):
    super(PixelDistanceLoss, self).__init__(reduction = reduction)
    self.globalBatchSize=globalBatchSize
    self.calculationType=calculationType
    self.normalizeDepthChannel=normalizeDepthChannel
    self.MAE_MSE = MAE_MSE

  def call(self, img1, img2):
    img1 = tf.cast(img1, tf.dtypes.float32)
    img2 = tf.cast(img2, tf.dtypes.float32)
    loss = 0.0

    if self.MAE_MSE == "mae":
      errorFunc = tf.abs
    elif self.MAE_MSE == "mse":
      errorFunc = tf.square
    else:
      raise Exception("parameter MAE_MSE={} is not defined. Use 'mae' or 'mse' instead.".format(self.norm))

    if self.calculationType == "per_channel":
      #initialize all weights with 1
      channelWeight = np.ones(img1.shape[-1])
      if self.normalizeDepthChannel:
        #set weight of the depth channel according to the number of color channels: e.g. for RGB = 3
        channelWeight[-1] = len(channelWeight)-1
        for i in range(img1.shape[-1]): 
          loss += channelWeight[i] * tf.reduce_mean(errorFunc(img1[:,:,:,i] - img2[:,:,:,i])) * (1. / self.globalBatchSize)

    elif self.calculationType == "per_image":
      loss = tf.reduce_mean(errorFunc(img1 - img2)) * (1. / self.globalBatchSize)

    else:
      raise Exception("pixel distance type is not defined: {}".format(self.calculationType))

    return loss
  
  
class StructuralSimilarityLoss(tf.keras.losses.Loss):
  '''
  calculates the structural similarity loss of two images of the shape (batch, height, width, channels)
  args:
    globalBatchSize: Batchsize of the training data to manually reduce the loss. When following a distributed strategy globalBatchSize = Num_worker * batchSizePerWorker
    calculationType (optional, default:"per_image"): determines whether the loss calculated per image or summed over all channels. Options: "per_image", "per_channel". 
    normalizeDepthChannel (optional, default: False): For RGBD images, the weight of depth is increased by multiplying the depth by the number of color channels.
    alpha (optional, default: 1): Weighting factor for contrast. 
    beta (optional, default: 1): Weighting factor for luminance.
    gamma (optional, default: 1): Weighting factor for structure. 
    c1 (optional, default: 0.0001): Constant considered in contrast calculation. 
    c2 (optional, default: 0.0009): Constant considered in luminance calculation.
  '''
  def __init__(self,globalBatchSize, calculationType = "per_image", normalizeDepthChannel = False, alpha = 1, beta = 1, gamma = 1, c1=0.0001, c2 =0.0009, reduction = tf.keras.losses.Reduction.AUTO):
    super(StructuralSimilarityLoss, self).__init__(reduction = reduction)
    self.globalBatchSize=globalBatchSize
    self.calculationType=calculationType
    self.normalizeDepthChannel=normalizeDepthChannel
    self.alpha = alpha
    self.beta = beta
    self.gamma = gamma
    self.c1 = c1
    self.c2 = c2
    self.c3 = c2/2

  def SSIM(self, tensor1, tensor2):
    mu1 = tf.reduce_mean(tensor1) #mean
    mu2 = tf.reduce_mean(tensor2)
    sigma1 = tf.reduce_mean((tensor1-mu1)**2)**0.5 #standard deviation
    sigma2 = tf.reduce_mean((tensor2-mu2)**2)**0.5  
    covar = tf.reduce_mean((tensor1-mu1)*(tensor2-mu2)) #covariance

    l = (2*mu1*mu2 + self.c1)/(mu1**2 + mu2**2 + self.c1)
    c = (2*sigma1*sigma2 + self.c2)/(sigma1**2 + sigma2**2 + self.c2)
    s = (covar + self.c3)/(sigma1 * sigma2 + self.c3)

    SSIM = l**self.alpha * c**self.beta * s**self.gamma 
    SSIM_neg = (1- SSIM) * (1. / self.globalBatchSize)

    return SSIM_neg

  def call(self, img1, img2):
    img1 = tf.cast(img1, tf.dtypes.float32)
    img2 = tf.cast(img2, tf.dtypes.float32)
    ssim = 0.0

    if self.calculationType == "per_image":
      ssim = self.SSIM(img1,img2)
    elif self.calculationType == "per_channel":
      #initialize all weights with 1
      channelWeight = np.ones(img1.shape[-1])
      if self.normalizeDepthChannel:
        #set weight of the depth channel according to the number of color channels: e.g. for RGB = 3
        channelWeight[-1] = len(channelWeight)-1
      #loop over all channels of the image
      for i in range(img1.shape[-1]):
        ssim += channelWeight[i]*self.SSIM(img1[:,:,:,i],img2[:,:,:,i])
    else:
      raise Exception("ssim calculation type is not defined")

    return ssim 
