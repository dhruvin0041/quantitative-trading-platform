import tensorflow as tf
import keras
from keras.layers import (
    Dense, 
    Dropout, 
    BatchNormalization, 
    MultiHeadAttention, 
    LayerNormalization, 
    Layer
)
from src.models.neural.lstm_branch import build_lstm_branch
from src.models.neural.cnn_branch import build_cnn_branch
from src.models.neural.transformer_branch import build_transformer_branch
from src.models.neural.tcn_agent import build_tcn_branch
from src.models.neural.patchtst_agent import build_patchtst_branch

class CrossModalAttention(Layer):
    """
    Dynamically weights the importance of different modality branches
    (LSTM, CNN, Transformer, TCN, PatchTST, Peer Context) depending on the market regime.
    """
    def __init__(self, num_heads=4, key_dim=64, **kwargs):
        super(CrossModalAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.attention = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
        self.layer_norm = LayerNormalization(epsilon=1e-6)
        
    def call(self, inputs):
        # inputs is a list of tensors of shape (batch, hidden_dim)
        # Stack them into (batch, num_modalities, hidden_dim)
        stacked = tf.stack(inputs, axis=1)
        
        # Self-attention across modalities
        attn_out, attention_scores = self.attention(
            query=stacked, 
            value=stacked, 
            key=stacked, 
            return_attention_scores=True
        )
        
        # Residual + Norm
        x = self.layer_norm(stacked + attn_out)
        
        # Average across modalities to get final representation
        fused = tf.reduce_mean(x, axis=1)
        return fused, attention_scores
        
    def get_config(self):
        config = super(CrossModalAttention, self).get_config()
        config.update({"num_heads": self.num_heads, "key_dim": self.key_dim})
        return config


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

    tcn_input, tcn_features = build_tcn_branch(
        time_steps=config["data"]["time_steps"],
        num_features=config["data"]["num_features"],
        nb_filters=m_config.get("tcn_filters", 64),
        dropout_rate=m_config.get("tcn_dropout", 0.2),
    )

    patchtst_input, patchtst_features = build_patchtst_branch(
        time_steps=config["data"]["time_steps"],
        num_features=config["data"]["num_features"],
        d_model=m_config.get("patch_d_model", 64),
        dropout=m_config.get("patch_dropout", 0.1),
    )

    # Lead-Lag Peer Context Branch
    peer_input, peer_features = build_lstm_branch(
        time_steps=config["data"]["time_steps"],
        num_features=config["data"]["num_features"],
        units_1=32,
        units_2=32,
        dropout_1=0.1,
        dropout_2=0.1,
        name="peer_context_data",
        out_name="peer_features"
    )
    
    # Force alignment of output dimensions to 64 for cross-attention
    lstm_aligned = Dense(64, activation="linear")(lstm_features)
    cnn_aligned = Dense(64, activation="linear")(cnn_features)
    transformer_aligned = Dense(64, activation="linear")(transformer_features)
    tcn_aligned = Dense(64, activation="linear")(tcn_features)
    patch_aligned = Dense(64, activation="linear")(patchtst_features)
    peer_aligned = Dense(64, activation="linear")(peer_features)

    # 2. Cross-Modal Attention Fusion (Replaces Concatenation)
    fused_features, attention_scores = CrossModalAttention(num_heads=4, key_dim=64)(
        [lstm_aligned, cnn_aligned, transformer_aligned, tcn_aligned, patch_aligned, peer_aligned]
    )

    combined = Dense(config["model"]["dense_units_1"], activation="relu")(fused_features)
    combined = BatchNormalization()(combined)
    combined = Dropout(config["model"]["dropout_rate"])(combined)

    combined = Dense(config["model"]["dense_units_2"], activation="relu")(combined)
    combined = BatchNormalization()(combined)

    # 3. Output Heads
    out_direction = Dense(1, activation="sigmoid", name="direction_output")(combined)
    out_range = Dense(2, activation="linear", name="range_output")(combined)
    out_signal = Dense(3, activation="softmax", name="signal_output")(combined)

    # 4. Compile Model
    model = keras.Model(
        inputs=[
            ts_input,
            cnn_input,
            transformer_input,
            tcn_input,
            patchtst_input,
            peer_input,
        ],
        outputs=[out_direction, out_range, out_signal],
    )

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=config["model"]["learning_rate"]
        ),
        loss={
            "direction_output": "binary_crossentropy",
            "range_output": "huber",
            "signal_output": "sparse_categorical_crossentropy",
        },
        loss_weights={
            "direction_output": 1.0,
            "range_output": 0.5,
            "signal_output": 1.0,
        },
        metrics={"direction_output": "accuracy", "signal_output": "accuracy"},
    )
    return model
