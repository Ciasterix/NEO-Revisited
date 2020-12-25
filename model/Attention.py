import tensorflow as tf


class Attention(tf.keras.layers.Layer):
    def __init__(self):
        super(Attention, self).__init__()
        self.dot = tf.keras.layers.Dot(axes=(1,2))

    def __call__(self, query, values):
        score = tf.expand_dims(self.dot([query, values]), -1)
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * values
        context_vector = tf.reduce_mean(context_vector, axis=1)

        return context_vector, attention_weights
