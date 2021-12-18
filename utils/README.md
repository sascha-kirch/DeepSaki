# Utils

# environment

## DetectHw

## EnableMixedPrecision
Mixed precision allows to set the preccison of the processing hardware. 
In contrast to just setting the data to a certain type, where the processing hardware still is able tu run at higher precission, whenn setting the processing hardware to e.g. float16, float32 opperations are not working anymore! Hence the code must be adapted acordingly.

Especially when using matplotlib, since it only supports float32 or higher

## EnableXlaAcceleration
XLA acceleration can be used to compile a graph for executions in order to accelerate the computation. 
This is manly for CPU and GPU, since the TPU dous compile the code as standard setting. When active, it takes some time in the first epoch of training, to compile the code.
