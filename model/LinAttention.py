import tensorflow as tf

from model.utils import masked_fill_


class LinAttention(tf.keras.layers.Layer):
    """
    Linear attention, as proposed by the linformer paper
    """

    def __init__(self, dim, dropout, E_proj, F_proj, full_attention=False):
        super(LinAttention, self).__init__()
        self.E = E_proj
        self.F = F_proj
        self.dim = dim
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.P_bar = None
        self.full_attention = full_attention
        self.is_proj_tensor = tf.is_tensor(E_proj)

    def __call__(self, Q, K, V, **kwargs):
        """
        Assume Q, K, V have same dtype
        E, F are `nn.Linear` modules
        """
        input_mask = kwargs["input_mask"] if "input_mask" in kwargs else None
        embeddings_mask = kwargs[
            "embeddings_mask"] if "embeddings_mask" in kwargs else None
        # Instead of classic masking, we have to do this, because the classic mask is of size nxn
        if input_mask is not None:
            # This is for k, v
            mask = input_mask[:, :, None]
            K = masked_fill_(K, tf.bitwise.invert(mask), 0.0)
            V = masked_fill_(V, tf.bitwise.invert(mask), 0.0)
            del mask

        if embeddings_mask is not None:
            mask = embeddings_mask[:, :, None]
            Q = masked_fill_(Q, tf.bitwise.invert(mask), 0.0)
            del mask

        K = tf.transpose(K, perm=[0, 2, 1])
        if not self.full_attention:
            if self.is_proj_tensor:
                K = tf.matmul(K, self.E)
            else:
                K = self.E(K)
        Q = tf.matmul(Q, K)

        P_bar = Q / tf.math.sqrt(tf.constant(self.dim, dtype=Q.dtype))
        P_bar = tf.nn.softmax(P_bar, axis=-1)

        P_bar = self.dropout(P_bar)

        if not self.full_attention:
            V = tf.transpose(V, perm=[0, 2, 1])
            if self.is_proj_tensor:
                V = tf.matmul(V, self.F)
            else:
                V = self.F(V)
            V = tf.transpose(V, perm=[0, 2, 1])
        out_tensor = tf.matmul(P_bar, V)

        return out_tensor
