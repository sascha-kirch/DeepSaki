# Detect hardware
from tensorflow.python.client import device_lib
import tensorflow as tf
from tensorflow.keras.mixed_precision import Policy as mixed_precision

def DetectHw(gpu_memory_groth = False):
  '''
  detects HW accelerators if present, initializes them and configures the distribution strategy
  args:
    gpu_memory_groth: Bool, if true and accelerator is GPU, memory groth is activated
  return:
    strategy: pointer to the distribution strategy configuration object
    runtime_environment: Info string describing the HW that might be used for conditions, i.e. "TPU, "GPU", "CPU"
    hw_accelerator_handle: pointer to the HW accelerator
  ''' 
  hw_accelerator_handle = None
  try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver() # TPU detection
    hw_accelerator_handle = tpu
  except ValueError:
    tpu = None
    if gpu_memory_groth:
      for gpu in tf.config.experimental.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    gpus = tf.config.experimental.list_logical_devices("GPU")
    hw_accelerator_handle = gpus
      
  # Select appropriate distribution strategy
  if tpu:
    runtime_environment = "TPU"
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu) # Going back and forth between TPU and host is expensive. Better to run 128 batches on the TPU before reporting back.
    print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])  
  elif len(gpus) > 1:
    runtime_environment = "GPU"
    strategy = tf.distribute.MirroredStrategy([gpu.name for gpu in gpus])
    print('Running on multiple GPUs ', [gpu.name for gpu in gpus])
  elif len(gpus) == 1:
    runtime_environment = "GPU"
    strategy = tf.distribute.get_strategy() # default strategy that works on CPU and single GPU
    print('Running on single GPU ', gpus[0].name)
  else:
    runtime_environment = "CPU"
    strategy = tf.distribute.get_strategy() # default strategy that works on CPU and single GPU
    print('Running on CPU')
  
  print("Number of accelerators: ", strategy.num_replicas_in_sync)
  print("____________________________________________________________________________________")
  print("Device List: ")
  print(device_lib.list_local_devices())

  return strategy, runtime_environment, hw_accelerator_handle


def EnableXlaAcceleration():
  '''
  Enables compiler acceleration for linear algebra
  '''
  tf.config.optimizer.set_jit(True)
  print('Linear algebra acceleration enabled')
  
  
def EnableMixedPrecision():
  '''
  Sets mixed precission depending on the presence of an HW accelerator. TPU uses bfloat format.
  '''
  try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver() # TPU detection
  except ValueError:
    tpu = None

  if tpu:
    policyConfig = 'mixed_bfloat16'
  else: 
    policyConfig = 'mixed_float16'
  policy = tf.keras.mixed_precision.Policy(policyConfig)
  tf.keras.mixed_precision.set_global_policy(policy)
  print('Mixed precision enabled to {}'.format(policyConfig))
