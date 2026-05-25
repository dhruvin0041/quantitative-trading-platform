import tensorflow as tf
from keras.layers import Input, Dense, Dropout, LayerNormalization, Add, Flatten, MultiHeadAttention, Lambda, Reshape
import keras

def build_patchtst_branch(time_steps, num_features, patch_len=12, stride=6, d_model=64, n_heads=4, e_layers=2, dropout=0.1):
    """
    Simplified implementation of PatchTST for time series forecasting.
    Channel-independence is key: each feature is treated as an independent 1D sequence, 
    patched, and passed through the transformer.
    """
    inputs = Input(shape=(time_steps, num_features), name="patchtst_input")
    
    # Calculate number of patches
    num_patches = int((time_steps - patch_len) / stride) + 1
    
    # Channel-independent patching
    # Transpose to (batch, num_features, time_steps)
    x = Lambda(lambda tt: tf.transpose(tt, perm=[0, 2, 1]))(inputs)
    
    # Create patches
    # Shape: (batch * num_features, num_patches, patch_len)
    def extract_patches_fn(tt):
        batch_size = tf.shape(tt)[0]
        tt_reshaped = tf.reshape(tt, [batch_size * num_features, time_steps, 1])
        patches = tf.image.extract_patches(
            images=tf.expand_dims(tt_reshaped, axis=-1),
            sizes=[1, patch_len, 1, 1],
            strides=[1, stride, 1, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )
        return tf.reshape(patches, [batch_size * num_features, num_patches, patch_len])

    x = Lambda(extract_patches_fn)(x)
    
    # Linear projection (embedding)
    x = Dense(d_model)(x)
    
    # Transformer Encoder blocks
    for _ in range(e_layers):
        attn_out = MultiHeadAttention(num_heads=n_heads, key_dim=d_model)(x, x)
        x = Add()([x, attn_out])
        x = LayerNormalization(epsilon=1e-6)(x)
        
        ffn = Dense(d_model * 4, activation='relu')(x) # Use relu for compatibility
        ffn = Dense(d_model)(ffn)
        x = Add()([x, ffn])
        x = LayerNormalization(epsilon=1e-6)(x)
        
    # Flatten the patches
    x = Flatten()(x)
    
    # Reshape back to combine features
    def reshape_back_fn(tt):
        batch_size = tf.shape(tt)[0] // num_features
        return tf.reshape(tt, [batch_size, num_features * num_patches * d_model])

    x = Lambda(reshape_back_fn)(x)
    
    # Final dense to output features
    x = Dense(128, activation='relu')(x)
    x = Dropout(dropout)(x)
    patchtst_features = Dense(64, activation='relu', name="patchtst_features")(x)
    
    return inputs, patchtst_features
