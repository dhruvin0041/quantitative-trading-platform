# src/features/sequence_builder.py
import logging

import numpy as np

logger = logging.getLogger(__name__)

try:
    import cupy as cp
    import torch
    HAS_CUPY = True if torch.cuda.is_available() else False
except ImportError:
    HAS_CUPY = False

def create_time_series_sequences(data_df, time_steps):
    """
    Converts a 2D Pandas DataFrame into a 3D NumPy array for LSTM input.
    Accelerated with CuPy/NumPy sliding window tricks.
    """
    # DATA LEAKAGE FIX: Explicitly exclude columns starting with 'target_' or 'future_'
    feature_columns = [
        col for col in data_df.columns
        if not col.startswith("target_") and not col.startswith("future_")
    ]

    data_array = data_df[feature_columns].values
    n_samples = len(data_array) - time_steps + 1

    if n_samples <= 0:
        return np.array([]), np.array([]), np.array([]), np.array([])

    # GPU Accelerated sliding window if CuPy is available
    if HAS_CUPY:
        try:
            cp_data = cp.asarray(data_array)
            shape = (n_samples, time_steps, data_array.shape[1])
            strides = (cp_data.strides[0], cp_data.strides[0], cp_data.strides[1])
            sequences = cp.lib.stride_tricks.as_strided(cp_data, shape=shape, strides=strides).get()
        except Exception as e:
            logger.warning("CuPy acceleration failed, falling back to NumPy: %s", e)
            shape = (n_samples, time_steps, data_array.shape[1])
            strides = (data_array.strides[0], data_array.strides[0], data_array.strides[1])
            sequences = np.lib.stride_tricks.as_strided(data_array, shape=shape, strides=strides)
    else:
        shape = (n_samples, time_steps, data_array.shape[1])
        strides = (data_array.strides[0], data_array.strides[0], data_array.strides[1])
        sequences = np.lib.stride_tricks.as_strided(data_array, shape=shape, strides=strides)

    sequences = sequences.copy()

    target_idx = np.arange(time_steps - 1, len(data_array))

    targets_direction = data_df["target_direction"].values[target_idx] if "target_direction" in data_df.columns else np.zeros(n_samples)
    targets_range_min = data_df["target_min"].values[target_idx] if "target_min" in data_df.columns else np.zeros(n_samples)
    targets_range_max = data_df["target_max"].values[target_idx] if "target_max" in data_df.columns else np.zeros(n_samples)

    return sequences, targets_direction, targets_range_min, targets_range_max
