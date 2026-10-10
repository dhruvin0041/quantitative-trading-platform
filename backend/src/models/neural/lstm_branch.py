# src/models/neural/lstm_branch.py
from keras.layers import (
    LSTM,
    BatchNormalization,
    Bidirectional,
    Dropout,
    Input,
)


def build_lstm_branch(
    time_steps,
    num_features,
    units_1=64,
    units_2=64,
    dropout_1=0.2,
    dropout_2=0.2,
    name="price_volume_data",
    out_name="lstm_features",
):
    """
    Unidirectional LSTM branch for standard sequential feature extraction.
    """
    ts_input = Input(shape=(time_steps, num_features), name=name)

    # LSTM Layer 1
    x = LSTM(units_1, return_sequences=True)(ts_input)
    x = BatchNormalization()(x)
    x = Dropout(dropout_1)(x)

    # LSTM Layer 2
    x = LSTM(units_2, return_sequences=False)(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_2)(x)

    ts_features = BatchNormalization(name=out_name)(x)

    return ts_input, ts_features


def build_bilstm_branch(
    time_steps,
    num_features,
    units_1=64,
    units_2=64,
    dropout_1=0.2,
    dropout_2=0.2,
    name="bilstm_input",
    out_name="bilstm_features",
):
    """
    Bi-directional LSTM (Long Short-Term Memory) branch for Sequential Feature Extraction.

    Role:
    - Captures deep historical context by processing sequences forward and backward
      across the historical window to identify momentum exhaustion and turning points.

    Zero Look-Ahead Bias Protocol:
    - In production live streaming feeds, the input tensor represents strictly causal
      historical data up to candle t-1 (or closed bar t).
    - The bidirectional sweep operates internally inside this historical window only,
      guaranteeing zero future leakage into the past.
    """
    ts_input = Input(shape=(time_steps, num_features), name=name)

    # Bi-LSTM Layer 1
    x = Bidirectional(LSTM(units_1, return_sequences=True))(ts_input)
    x = BatchNormalization()(x)
    x = Dropout(dropout_1)(x)

    # Bi-LSTM Layer 2
    x = Bidirectional(LSTM(units_2, return_sequences=False))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_2)(x)

    ts_features = BatchNormalization(name=out_name)(x)

    return ts_input, ts_features

