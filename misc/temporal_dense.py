"""
This Module implements the TemporalDense layer.

This layer processes input of the form [batch, time, dim] along the time dimension.

It applies a dense feedforward computation along the time axis,
processing along one dimension of the object dimensions at a time.
"""

import tensorflow as tf

from keras.engine.base_layer import Layer

from keras import activations
from keras import backend
from keras import constraints
from keras import initializers
from keras import regularizers

from keras.engine.input_spec import InputSpec

class TemporalDense(Layer):

    def __init__(
        self,
        units = None,
        activation=None,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
        kernel_regularizer=None,
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        **kwargs,
    ):
        """create TemporalDense layer.

        Processes input of shape [time, object_dim] along the time axis
        via a feedforward dense operation. This is done individually along
        each dimension of the object vectors.

        Parameters
        ----------
        units : int, optional
            number of temporal units, by default None
        activation : string or tensorflow activation function, optional
            activation function. 'linear' if not specified, by default None
        use_bias : bool, optional
            whether to add a (temporal) bias vector, by default True
        kernel_initializer : str, optional
            initializer for `kernel` weights matrix, by default "glorot_uniform"
        bias_initializer : str, optional
            initializer for `bias` weights vector, by default "zeros"
        kernel_regularizer : tensorflow regularizer, optional
            regualizer to use on `kernel` weights, by default None
        bias_regularizer : tensorflow regularizer, optional
            regualizer to use on `bias` weights, by default None
        activity_regularizer : tenosrflow regularizer, optional
            regularizer to use on the output/activation of layer, by default None
        kernel_constraint : tensorflow constraint function, optional
            constraint to apply to `kernel` weights matrix, by default None
        bias_constraint : tensorflow constraint function, optional
            constraint to apply to `bias` weights vector, by default None
        """

        super().__init__(activity_regularizer=activity_regularizer, **kwargs)

        self.units = int(units) if (not isinstance(units, int) and units is not None) else units
        self.activation = activations.get(activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.supports_masking = True

    def build(self, input_shape):

        dtype = tf.as_dtype(self.dtype or backend.floatx())
        if not (dtype.is_floating or dtype.is_complex):
            raise TypeError(
                "A Dense layer can only be built with a floating-point "
                f"dtype. Received: dtype={dtype}"
            )

        batch_dim, temporal_dim, object_dim = input_shape
        input_shape = tf.TensorShape(input_shape)

        # if units is None, default to the temporal dimension
        if self.units is None:
            self.units = temporal_dim

        # create input spec: minimum # of dimensions is 3 (batch, time, object_dim)
        # temporal dimension needs to be the same for all inputs
        self.input_spec = InputSpec(min_ndim=3, axes={1: temporal_dim})

        # create weights
        self.kernel = self.add_weight(
            "kernel",
            shape=[temporal_dim, self.units],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True,
        )

        if self.use_bias:
            self.bias = self.add_weight(
                "bias",
                shape=[
                    self.units,1
                ],
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                dtype=self.dtype,
                trainable=True,
            )
        else:
            self.bias = None

        self.built = True

    def call(self, inputs):

        # cast type if necessary
        if inputs.dtype.base_dtype != self._compute_dtype_object.base_dtype:
            inputs = tf.cast(inputs, dtype=self._compute_dtype_object)

        # apply kernel along time axis
        outputs = tf.einsum('tn,btd->bnd', self.kernel, inputs)

        # add bias
        if self.use_bias:
            outputs = outputs + self.bias

        # apply activation
        if self.activation is not None:
            outputs = self.activation(outputs)

        return outputs

    def compute_output_shape(self, input_shape):
        batch_dim, temporal_dim, object_dim = input_shape

        output_shape = tf.TensorShape((batch_dim, self.units, object_dim))

        return output_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "units": self.units,
                "activation": activations.serialize(self.activation),
                "use_bias": self.use_bias,
                "kernel_initializer": initializers.serialize(
                    self.kernel_initializer
                ),
                "bias_initializer": initializers.serialize(
                    self.bias_initializer
                ),
                "kernel_regularizer": regularizers.serialize(
                    self.kernel_regularizer
                ),
                "bias_regularizer": regularizers.serialize(
                    self.bias_regularizer
                ),
                "activity_regularizer": regularizers.serialize(
                    self.activity_regularizer
                ),
                "kernel_constraint": constraints.serialize(
                    self.kernel_constraint
                ),
                "bias_constraint": constraints.serialize(self.bias_constraint),
            }
        )
        return config