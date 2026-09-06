import numpy as np
import tensorflow as tf
from keras.layers import (
    Add,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    Input,
    Layer,
    LayerNormalization,
    MultiHeadAttention,
)


class PositionalEncoding(Layer):
    """
    Sinusoidal Positional Encoding for Transformer to preserve sequence order.
    """

    def __init__(self, max_steps, max_dims, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.max_steps = max_steps
        self.max_dims = max_dims
        if max_dims % 2 == 1:
            max_dims += 1  # Ensure even for sin/cos pairs

        p, i = np.meshgrid(np.arange(max_steps), np.arange(max_dims // 2))
        pos_emb = np.empty((1, max_steps, max_dims))
        pos_emb[0, :, ::2] = np.sin(p / 10000 ** (2 * i / max_dims)).T
        pos_emb[0, :, 1::2] = np.cos(p / 10000 ** (2 * i / max_dims)).T
        self.positional_encoding = tf.constant(pos_emb, dtype=tf.float32)

    def call(self, inputs):
        shape = tf.shape(inputs)
        return inputs + self.positional_encoding[:, : shape[1], : shape[2]]

    def get_config(self):
        config = super(PositionalEncoding, self).get_config()
        config.update({"max_steps": self.max_steps, "max_dims": self.max_dims})
        return config


def build_transformer_branch(
    time_steps,
    num_features,
    head_size=128,
    num_heads=4,
    ff_dim=128,
    dropout=0.1,
    positional_encoding=True,
):
    """
    Transformer branch for capturing complex long-range dependencies in time-series data.
    """
    inputs = Input(shape=(time_steps, num_features), name="transformer_input")

    x = inputs
    if positional_encoding:
        x = PositionalEncoding(max_steps=time_steps, max_dims=num_features)(x)

    # Transformer Encoder Block
    # 1. Multi-Head Attention
    attention_output = MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = Add()([x, attention_output])  # Residual Connection
    x = LayerNormalization(epsilon=1e-6)(x)

    # 2. Feed Forward Network
    ffn = Dense(ff_dim, activation="relu")(x)
    ffn = Dropout(dropout)(ffn)
    ffn = Dense(num_features)(ffn)
    x = Add()([x, ffn])  # Residual Connection
    x = LayerNormalization(epsilon=1e-6)(x)

    # Output
    x = GlobalAveragePooling1D(data_format="channels_last")(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.1)(x)
    transformer_features = Dense(64, activation="relu", name="transformer_features")(x)

    return inputs, transformer_features
