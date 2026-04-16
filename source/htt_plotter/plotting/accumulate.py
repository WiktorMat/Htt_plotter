import numpy as np


def add_histogram(container: dict, sample: str, counts: np.ndarray) -> None:
    if sample not in container:
        container[sample] = np.zeros_like(counts, dtype=float)

    container[sample] += counts
