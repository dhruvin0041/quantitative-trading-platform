import keras
import tensorflow as tf
from keras.layers import (
    LSTM,
    Add,
    Concatenate,
    Dense,
    Dropout,
    Input,
    Lambda,
    LayerNormalization,
    MultiHeadAttention,
    Multiply,
)


class GatedResidualNetwork(keras.layers.Layer):
    def __init__(self, hidden_dim, dropout_rate=0.1, **kwargs):
        super(GatedResidualNetwork, self).__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate

        self.dense1 = Dense(hidden_dim, activation="elu")
        self.dense2 = Dense(hidden_dim)
        self.dropout = Dropout(dropout_rate)

        # GLU (Gated Linear Unit)
        self.glu_dense1 = Dense(hidden_dim)
        self.glu_dense2 = Dense(hidden_dim, activation="sigmoid")

        self.layer_norm = LayerNormalization()

    def call(self, inputs):
        # Skip connection if dimensions match
        x = inputs

        # ELU + Dense
        out = self.dense1(x)
        out = self.dense2(out)
        out = self.dropout(out)

        # GLU
        gate = self.glu_dense2(out)
        val = self.glu_dense1(out)
        glu_out = Multiply()([gate, val])

        # Residual + Norm
        # If input dim != hidden_dim, we'd need a projection here in a full implementation
        return self.layer_norm(Add()([x, glu_out]))

    def get_config(self):
        config = super(GatedResidualNetwork, self).get_config()
        config.update(
            {"hidden_dim": self.hidden_dim, "dropout_rate": self.dropout_rate}
        )
        return config


def total_quantile_loss(quantiles):
    def loss(y_true, y_pred):
        # y_true shape: (batch, 1)
        # y_pred shape: (batch, 5)
        y_true = tf.cast(y_true, tf.float32)
        if len(y_true.shape) == 1:
            y_true = tf.expand_dims(y_true, axis=-1)

        q_losses = []
        for i, q in enumerate(quantiles):
            y_p = y_pred[:, i : i + 1]
            e = y_true - y_p
            q_losses.append(tf.maximum(q * e, (q - 1) * e))

        return tf.reduce_mean(tf.add_n(q_losses))

    return loss


def build_tft_branch(time_steps, num_features, hidden_dim=64, num_heads=4, dropout=0.1):
    """
    Simplified Temporal Fusion Transformer (TFT) branch.
    Focuses on Gated Residual Networks (GRNs) and sequence-to-sequence attention.
    """
    inputs = Input(shape=(time_steps, num_features), name="tft_input")

    # 1. Variable Selection Network (Simplified as a GRN per timestep)
    # Project inputs to hidden dimension
    x = Dense(hidden_dim)(inputs)
    x = GatedResidualNetwork(hidden_dim, dropout_rate=dropout)(x)

    # 2. Local Processing (LSTM Encoder)
    # TFT uses LSTMs to generate local context
    lstm_out = LSTM(hidden_dim, return_sequences=True)(x)

    # Gate and add residual
    gate = Dense(hidden_dim, activation="sigmoid")(lstm_out)
    gated_lstm = Multiply()([lstm_out, gate])
    x = LayerNormalization()(Add()([x, gated_lstm]))

    # 3. Temporal Attention
    # Multi-head attention across time
    attn_out = MultiHeadAttention(num_heads=num_heads, key_dim=hidden_dim)(x, x)

    # Gate and add residual
    gate_attn = Dense(hidden_dim, activation="sigmoid")(attn_out)
    gated_attn = Multiply()([attn_out, gate_attn])
    x = LayerNormalization()(Add()([x, gated_attn]))

    # 4. Final Processing
    x = GatedResidualNetwork(hidden_dim, dropout_rate=dropout)(x)

    # Aggregate over time
    hidden = Lambda(lambda tt: tf.reduce_mean(tt, axis=1))(x)

    # Quantile outputs for probabilistic forecasting [0.1, 0.25, 0.5, 0.75, 0.9]
    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    outputs = []
    for q in quantiles:
        q_output = Dense(1, name=f"quantile_{int(q * 100)}")(hidden)
        outputs.append(q_output)

    # Concatenate quantile outputs along a new axis
    # Resulting shape: (batch, 5)
    output = Concatenate(name="quantile_output")(outputs)

    return inputs, output
