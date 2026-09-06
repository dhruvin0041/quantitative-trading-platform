# src/models/lstm_branch.py
from keras.layers import LSTM, BatchNormalization, Dropout, Input


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
