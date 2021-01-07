import tensorflow as tf


def get_EF(input_size, dim, method="learnable", head_dim=None, bias=True):
    """
    Retuns the E or F matrix, initialized via xavier initialization.
    This is the recommended way to do it according to the authors of the paper.
    Includes a method for convolution, as well as a method for no additional params.
    """
    assert method == "learnable" or method == "convolution" or method == "no_params", "The method flag needs to be either 'learnable', 'convolution', or 'no_params'!"
    if method == "convolution":
        conv = tf.keras.layers.Conv1D(filters=head_dim,
                                      kernel_size=int(input_size / dim),
                                      strides=int(input_size / dim))
        return conv
    if method == "no_params":
        init = tf.random_normal_initializer(mean=0.0, stddev=1 / dim)
        mat = tf.Variable(init(shape=[input_size, dim]), trainable=False)
        return mat
    init = tf.random_normal_initializer(mean=0.0, stddev=1 / dim)
    lin = tf.keras.layers.Dense(units=dim,
                                use_bias=bias,
                                kernel_initializer="glorot_normal")
    return lin


def masked_fill_(tensor, mask, value):
    return tf.where(mask, value, tensor)
