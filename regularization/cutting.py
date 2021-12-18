import tensorflow as tf        
import numpy as np             

def _random_boundingbox(height, width, lam):
  '''Generates a random bounding box'''
  r = np.sqrt(1. - lam)
  w = np.int(width * r)
  h = np.int(height * r)
  x = np.random.randint(width)
  y = np.random.randint(height)

  x1 = np.clip(x - w // 2, 0, width)
  y1 = np.clip(y - h // 2, 0, height)
  x2 = np.clip(x + w // 2, 0, width)
  y2 = np.clip(y + h // 2, 0, height)

  return x1, y1, x2, y2

def GetMask(maskShape):
  '''generates a mask for a given image dimensions (batch_size, height, width,channel)'''
  mask = np.ones(shape=maskShape)
  for element in range(maskShape[0]):
    lam = np.random.beta(1,1)
    x1, y1, x2, y2 = _random_boundingbox(maskShape[1], maskShape[2], lam)
    mask[element, x1:x2,y1:y2,:]=0
  mask = tf.convert_to_tensor(mask, dtype=tf.float32)
  return mask

def CutMix(batch1, batch2, ignoreBackground = False, invert_mask = False, mask = None):
  '''
  performs the cutmix operation of two image batches
  args:
    batch1: batch of grid-shaped data with shape (batch, height, width, channel)
    batch2: batch of grid-shaped data with shape (batch, height, width, channel)
    ignoreBackground: bool to indicate whether or not to ignore 0-valued backgrounds
    invert_mask: bool to indicate whether or not the mask is inverted
    mask: optional mask can be provided. If "None", mask is generated
  
  return:
    ground_truth_mask: actual mask that has been applied
    new_batch: batch with applied coutmix opperation
  '''
  
  batch1 = tf.cast(batch1,tf.float32)
  batch2 = tf.cast(batch2,tf.float32)
  
  if mask == None: # generate mask
    mask = GetMask(maskShape=batch1.shape)
    
  if ignoreBackground: #check where in image are no background pixels (value = 1)
    batch1_mask = tf.cast(tf.where(batch1 > 0 ,1 , 0), tf.int32)
    batch2_mask = tf.cast(tf.where(batch2 > 0 ,1 , 0), tf.int32)
    mutal_person_mask = tf.cast(tf.clip_by_value((batch1_mask + batch2_mask),0,1), tf.float32)    
    ground_truth_mask = (1-(1-mask) * mutal_person_mask)

  else:
    ground_truth_mask = mask

  if invert_mask:
      ground_truth_mask = 1 - ground_truth_mask

  new_batch = batch1*ground_truth_mask + batch2*(1-ground_truth_mask)

  return ground_truth_mask, new_batch

def CutOut(batch, invert_mask = False, mask = None):
  '''
  performs the cutout operation on an input batch
  args:
    batch: batch of grid-shaped data with shape (batch, height, width, channel)
    invert_mask: bool to indicate whether or not the mask is inverted
    mask: optional mask can be provided. If "None", mask is generated
  
  return:
    mask: actual mask that has been applied
    new_batch: batch with applied coutout opperation
  '''
  
  batch = tf.cast(batch,tf.float32)
  
  if mask == None: # generate mask
    mask = GetMask(maskShape=batch.shape)

  if invert_mask:
    mask = 1 - mask

  new_batch = batch*mask
  return mask, new_batch
