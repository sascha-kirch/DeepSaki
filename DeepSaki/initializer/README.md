# Initializer
This package contains several initializers. They can be accessed by 
```python
import DeepSaki
initializer = DeepSaki.initializer.<INITIALIZER>
```

## HeAlphaNormal
This initializer is based on the [He initializer](https://arxiv.org/abs/1502.01852). 
In contrast to the tensorflow implementation, an alpha value can be set to consider the non-zero slope of a LeakyReLU activation.

The weights of a certain layer W[l] are drawn from a normal distribution N(...):
```
W[l] ~ N(µ = 0, σ²= 2/((1+α²)n[l])
```
where µ is the mean, σ² is the variance, α is a configurable variable and n[l] is the number of parameters in layer 𝑙

## HeAlphaUniform
The weights of a certain layer W[l] are drawn from a uniform distribution U(...):
```
W[l] ~ U(a = -sqrt(6/((1+α²)n[l])), b= sqrt(6/((1+α²)n[l])))
```
where α is a configurable variable and n[l] is the number of parameters in layer 𝑙
