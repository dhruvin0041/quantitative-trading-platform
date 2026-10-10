import unittest

from src.models.neural.lstm_branch import build_bilstm_branch, build_lstm_branch


class TestBiLSTMBranch(unittest.TestCase):
    def test_build_bilstm_branch_shape_and_layers(self):
        time_steps = 30
        num_features = 10
        units_1 = 32
        units_2 = 16

        ts_input, ts_features = build_bilstm_branch(
            time_steps=time_steps,
            num_features=num_features,
            units_1=units_1,
            units_2=units_2,
        )

        self.assertEqual(ts_input.shape[1:], (time_steps, num_features))
        # For Bidirectional LSTM with units_2, forward + backward = 2 * units_2
        self.assertEqual(ts_features.shape[1:], (2 * units_2,))

    def test_build_unidirectional_lstm_branch(self):
        time_steps = 20
        num_features = 8
        units_1 = 32
        units_2 = 16

        ts_input, ts_features = build_lstm_branch(
            time_steps=time_steps,
            num_features=num_features,
            units_1=units_1,
            units_2=units_2,
        )

        self.assertEqual(ts_input.shape[1:], (time_steps, num_features))
        self.assertEqual(ts_features.shape[1:], (units_2,))


if __name__ == "__main__":
    unittest.main()
