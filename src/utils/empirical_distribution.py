import numpy as np


class EDF:
    observations: np.ndarray
    edf: np.ndarray
    max_value: float
    min_value: float

    def __init__(self, observations: np.ndarray, min_value: float, max_value: float = None):
        if max_value is not None and max_value > np.max(observations):
            observations = np.append(observations, max_value)
        assert min_value <= np.min(observations)
        self.observations, counts = np.unique(observations, return_counts=True)
        self.edf = np.cumsum(counts) / len(observations)
        self.max_value = max_value
        self.min_value = min_value

    def sample(self, num_samples: int, seed=None):
        rng = np.random.default_rng(seed)
        uni_samples = rng.uniform(0, 1, size=num_samples)
        sample_index = np.searchsorted(self.edf, uni_samples, side='right')

        zero_indices_mask = sample_index == 0
        num_zero = np.sum(zero_indices_mask)
        in_range_indices = sample_index[~zero_indices_mask]

        start_values = self.observations[in_range_indices - 1]
        end_values = self.observations[in_range_indices]
        values = start_values + rng.uniform(0, 1, size=len(start_values)) * (end_values - start_values)

        min_values = self.min_value + rng.uniform(0, 1, size=num_zero) * (self.observations[0] - self.min_value)

        results = rng.permutation(np.concatenate((values, min_values), axis=0))
        return results
