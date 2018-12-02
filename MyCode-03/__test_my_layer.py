from tensorflow.keras import layers
from tensorflow.keras import activations
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer

class MyLayerV1(Layer):

    def __init__(self, units, activation=None, **kwargs):
        self.output_dim = units
        self.activation = activations.get(activation)
        super(MyLayerV1, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        input_dim = input_shape[1:].as_list()
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(self.output_dim, *input_dim),
                                      initializer='glorot_uniform',
                                      trainable=True)
        super(MyLayerV1, self).build(input_shape)

    def call(self, x):
        x = K.expand_dims(x, axis=1)
        output = K.sum(x * self.kernel, axis=(2, 3))
        if self.activation is not None:
            return self.activation(output)
        return output

    def compute_output_shape(self, input_shape):
        return (None, self.output_dim, input_shape[-1])

if __name__ == "__main__":
    MyLayerV1(120, 'sigmoid')(layers.Input(shape=(28, 28, 32)))