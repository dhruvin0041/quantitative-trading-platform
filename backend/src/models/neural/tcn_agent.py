import tensorflow as tf
from keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Conv1D,
    Dense,
    Input,
    SpatialDropout1D,
)


def build_residual_block(
    x, dilation_rate, nb_filters, kernel_size, padding, dropout_rate=0.2
):
    """
    A residual block for Temporal Convolutional Network (TCN).
    """
    prev_x = x

    # First conv
    x = Conv1D(
        filters=nb_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding=padding,
    )(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = SpatialDropout1D(rate=dropout_rate)(x)

    # Second conv
    x = Conv1D(
        filters=nb_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding=padding,
    )(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = SpatialDropout1D(rate=dropout_rate)(x)

    # 1x1 conv to match shapes if necessary
    if prev_x.shape[-1] != nb_filters:
        prev_x = Conv1D(nb_filters, 1, padding="same")(prev_x)

    res_x = Add()([prev_x, x])
    return Activation("relu")(res_x)


def build_tcn_branch(
    time_steps,
    num_features,
    nb_filters=64,
    kernel_size=3,
    dilations=[1, 2, 4, 8, 16],
    dropout_rate=0.2,
):
    """
    Temporal Convolutional Network (TCN) branch for time series pattern extraction.
    """
    inputs = Input(shape=(time_steps, num_features), name="tcn_input")
    x = inputs

    for dilation_rate in dilations:
        x = build_residual_block(
            x,
            dilation_rate=dilation_rate,
            nb_filters=nb_filters,
            kernel_size=kernel_size,
            padding="causal",
            dropout_rate=dropout_rate,
        )

    # Extract features
    # Since causal padding is used, the last timestep contains info about the whole sequence
    x = tf.keras.layers.Lambda(lambda tt: tt[:, -1, :])(x)
    tcn_features = Dense(64, activation="relu", name="tcn_features")(x)

    return inputs, tcn_features
