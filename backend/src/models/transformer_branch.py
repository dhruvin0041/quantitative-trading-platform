from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    MultiHeadAttention,
    LayerNormalization,
    GlobalAveragePooling1D,
    Add,
)


def build_transformer_branch(
    time_steps, num_features, head_size=128, num_heads=4, ff_dim=128, dropout=0.1
):
    """
    Transformer branch for capturing complex long-range dependencies in time-series data.
    """
    inputs = Input(shape=(time_steps, num_features), name="transformer_input")

    # Simple Positional Encoding (Optional: can be expanded)
    x = inputs

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
