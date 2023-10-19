import tensorflow as tf

def MakeInitializerComplex(initializer:tf.keras.initializers.Initializer):
    def ComplexInitializer(shape,dtype):
        if dtype == tf.complex64:
            dtype = tf.float32
        elif dtype == tf.complex128:
            dtype = tf.float64
        real = initializer(shape,dtype)
        imag = initializer(shape,dtype)
        return tf.dtypes.complex(real,imag)
    return ComplexInitializer