# src/features/sequence_builder.py
import numpy as np


def create_time_series_sequences(data_df, time_steps):
    """
    Converts a 2D Pandas DataFrame into a 3D NumPy array for LSTM input.
    """
    sequences = []
    targets_direction = []
    targets_range_min = []
    targets_range_max = []

    # Assuming the dataframe has pre-calculated features and future targets
    # DATA LEAKAGE FIX: Explicitly exclude columns starting with 'target_' or 'future_'
    feature_columns = [
        col
        for col in data_df.columns
        if not col.startswith("target_") and not col.startswith("future_")
    ]

    data_array = data_df[feature_columns].values

    for i in range(len(data_array) - time_steps + 1):
        # Create rolling window of features
        seq = data_array[i : (i + time_steps)]
        sequences.append(seq)

        # Extract targets for the current step (last element in the sequence)
        targets_direction.append(data_df["target_direction"].iloc[i + time_steps - 1])
        targets_range_min.append(data_df["target_min"].iloc[i + time_steps - 1])
        targets_range_max.append(data_df["target_max"].iloc[i + time_steps - 1])

    return (
        np.array(sequences),
        np.array(targets_direction),
        np.array(targets_range_min),
        np.array(targets_range_max),
    )
