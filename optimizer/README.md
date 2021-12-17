# Optimizer
This package contains several optimizers. They can be accessed by 
```python
import DeepSaki
optimizer = DeepSaki.optimizer.<OPTIMIZER>
```
  
## SWATS_ADAM
SWATS_ADAM is inspired by the SWATS (switching from adam to sgd) initializer (see [HERE](http://arxiv.org/abs/1712.07628)).
During training, the optimizer can be changed by modifying the attribute ```self.currentOptimizer``` to either ```"adam"``` or ```"sgd"```.

This optimizer combines tensorflow's [ADAM](https://github.com/tensorflow/tensorflow/blob/85c8b2a817f95a3e979ecd1ed95bff1dc1335cff/tensorflow/python/keras/optimizer_v2/adam.py) and [SGD](https://github.com/tensorflow/tensorflow/blob/85c8b2a817f95a3e979ecd1ed95bff1dc1335cff/tensorflow/python/keras/optimizer_v2/gradient_descent.py) implementations.

**Example:**
```python
import DeepSaki
optimizer = DeepSaki.optimizer.SWATS_ADAM(learning_rate=disc_learning_rate, beta_1=0.5, amsgrad=False, momentum=0.9, nesterov=False)
optimizer.currentOptimizer = "sgd"
```


## SWATS_NADAM
SWATS_NADAM is inspired by the SWATS (switching from adam to sgd) initializer (see [HERE](http://arxiv.org/abs/1712.07628)).
During training, the optimizer can be changed by modifying the attribute ```self.currentOptimizer``` to either ```"nadam"``` or ```"sgd"```.

This optimizer combines tensorflow's [NADAM](https://github.com/tensorflow/tensorflow/blob/85c8b2a817f95a3e979ecd1ed95bff1dc1335cff/tensorflow/python/keras/optimizer_v2/nadam.py) and [SGD](https://github.com/tensorflow/tensorflow/blob/85c8b2a817f95a3e979ecd1ed95bff1dc1335cff/tensorflow/python/keras/optimizer_v2/gradient_descent.py) implementations.

**Example:**
```python
import DeepSaki
optimizer = DeepSaki.optimizer.SWATS_NADAM(learning_rate=disc_learning_rate, beta_1=0.5, momentum=0.9, nesterov=False)
optimizer.currentOptimizer = "nadam"
```
