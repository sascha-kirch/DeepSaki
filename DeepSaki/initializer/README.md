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

$$W^{[l]} \sim \mathcal{N}\left(\mu = 0, \sigma^{2}= \frac{2}{(1+\alpha^{2})n[l]}\right)$$

where µ is the mean, σ² is the variance, α is a configurable variable and n[l] is the number of parameters in layer 𝑙

## HeAlphaUniform
The weights of a certain layer W[l] are drawn from a uniform distribution U(...):

$$W^{[l]} \sim \mathcal{U}\left(a = -\sqrt{\frac{6}{n^{[l]}+n^{[l+1]}}}, b = \sqrt{\frac{6}{n^{[l]}+n^{[l+1]}}}\right)$$

where α is a configurable variable and n[l] is the number of parameters in layer 𝑙
