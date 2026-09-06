from keras.layers import (
    BatchNormalization,
    Conv1D,
    Dense,
    Dropout,
    Flatten,
    Input,
    MaxPooling1D,
)


def build_cnn_branch(
    time_steps, num_features, filters_1=32, filters_2=64, kernel_size=3, dense_units=64
):
    """
    CNN branch for pattern recognition in the recent price/indicator matrix.
    Input: last 30-60 candles as an image-like matrix.
    """
    cnn_input = Input(shape=(time_steps, num_features), name="cnn_input")

    # 1D Convolutional Layers (treating time-series as 1D image)
    x = Conv1D(
        filters=filters_1, kernel_size=kernel_size, activation="relu", padding="same"
    )(cnn_input)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = Conv1D(
        filters=filters_2, kernel_size=kernel_size, activation="relu", padding="same"
    )(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = Flatten()(x)
    x = Dense(dense_units, activation="relu")(x)
    x = Dropout(0.2)(x)
    cnn_features = BatchNormalization(name="cnn_features")(x)

    return cnn_input, cnn_features
