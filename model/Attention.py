import tensorflow as tf


class Attention(tf.keras.layers.Layer):
    def __init__(self):
        super(Attention, self).__init__()

    def __call__(self, query, values):
        key = values
        score = tf.matmul(query, key, transpose_b=True)
        attention_weights = tf.nn.softmax(score)
        context_vector = tf.matmul(attention_weights, values)

        return context_vector, attention_weights
