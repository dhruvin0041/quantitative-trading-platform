# src/models/finbert_branch.py
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda
from transformers import TFBertModel

# Load pre-trained FinBERT ONCE at module level to save memory and speed up optimization
_finbert_model = None


def get_finbert():
    global _finbert_model
    if _finbert_model is None:
        print("Loading FinBERT into memory...")
        _finbert_model = TFBertModel.from_pretrained("ProsusAI/finbert")
        _finbert_model.trainable = False
    return _finbert_model


def build_finbert_branch(max_seq_length, dense_units=64):
    text_input_ids = Input(
        shape=(max_seq_length,), dtype=tf.int32, name="news_input_ids"
    )
    text_attention_mask = Input(
        shape=(max_seq_length,), dtype=tf.int32, name="news_attention_mask"
    )

    finbert = get_finbert()

    # Use a Lambda layer to wrap the BERT call.
    # This prevents Keras 3 from complaining about KerasTensor vs TensorFlow Tensor types.
    bert_output = Lambda(
        lambda x: finbert(input_ids=x[0], attention_mask=x[1])[1], output_shape=(768,)
    )([text_input_ids, text_attention_mask])

    sentiment_features = Dense(
        dense_units, activation="relu", name="sentiment_features"
    )(bert_output)

    return text_input_ids, text_attention_mask, sentiment_features
