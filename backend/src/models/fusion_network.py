# src/models/fusion_network.py
import tensorflow as tf
from tensorflow.keras.layers import Concatenate, Dense, Dropout, BatchNormalization
from src.models.lstm_branch import build_lstm_branch
from src.models.cnn_branch import build_cnn_branch
from src.models.transformer_branch import build_transformer_branch
from src.models.finbert_branch import build_finbert_branch


def build_fusion_model(config):
    m_config = config.get("model", {})

    # 1. Build Branches with Granular Config
    ts_input, lstm_features = build_lstm_branch(
        time_steps=config["data"]["time_steps"],
        num_features=config["data"]["num_features"],
        units_1=m_config.get("lstm_units_1", 64),
        units_2=m_config.get("lstm_units_2", 64),
        dropout_1=m_config.get("lstm_dropout_1", 0.2),
        dropout_2=m_config.get("lstm_dropout_2", 0.2),
    )

    cnn_input, cnn_features = build_cnn_branch(
        time_steps=config["data"]["time_steps"],
        num_features=config["data"]["num_features"],
        filters_1=m_config.get("cnn_filters_1", 32),
        filters_2=m_config.get("cnn_filters_2", 64),
        kernel_size=m_config.get("cnn_kernel", 3),
        dense_units=m_config.get("cnn_dense", 64),
    )

    transformer_input, transformer_features = build_transformer_branch(
        time_steps=config["data"]["time_steps"],
        num_features=config["data"]["num_features"],
        head_size=m_config.get("trans_head_size", 128),
        num_heads=m_config.get("trans_heads", 4),
        ff_dim=m_config.get("trans_ff_dim", 128),
        dropout=m_config.get("trans_dropout", 0.1),
    )

    # NEW: Lead-Lag Peer Context Branch
    peer_input, peer_features = build_lstm_branch(
        time_steps=config["data"]["time_steps"],
        num_features=config["data"]["num_features"],
        units_1=32,
        units_2=32,
        dropout_1=0.1,
        dropout_2=0.1,
    )

    text_input_ids, text_attention_mask, sentiment_features = build_finbert_branch(
        max_seq_length=config["data"]["max_seq_length"]
    )

    # 2. Fuse Features from ALL branches
    combined = Concatenate()(
        [
            lstm_features,
            cnn_features,
            transformer_features,
            peer_features,
            sentiment_features,
        ]
    )

    combined = Dense(config["model"]["dense_units_1"], activation="relu")(combined)
    combined = BatchNormalization()(combined)
    combined = Dropout(config["model"]["dropout_rate"])(combined)

    combined = Dense(config["model"]["dense_units_2"], activation="relu")(combined)
    combined = BatchNormalization()(combined)

    # 3. Output Heads
    out_direction = Dense(1, activation="sigmoid", name="direction_output")(combined)
    out_range = Dense(2, activation="linear", name="range_output")(combined)
    out_signal = Dense(3, activation="softmax", name="signal_output")(combined)

    # 4. Compile Model
    model = tf.keras.Model(
        inputs=[
            ts_input,
            cnn_input,
            transformer_input,
            peer_input,
            text_input_ids,
            text_attention_mask,
        ],
        outputs=[out_direction, out_range, out_signal],
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=config["model"]["learning_rate"]
        ),
        loss={
            "direction_output": "binary_crossentropy",
            "range_output": "huber_loss",
            "signal_output": "sparse_categorical_crossentropy",
        },
        loss_weights={
            "direction_output": 1.0,
            "range_output": 0.5,  # Give less weight to range regression
            "signal_output": 1.0,
        },
        metrics={"direction_output": "accuracy", "signal_output": "accuracy"},
    )
    return model
