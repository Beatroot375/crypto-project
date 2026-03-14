from __future__ import annotations

import unittest

import numpy as np

from quant_crypto_portfolio.drift import population_stability_index


class TestDrift(unittest.TestCase):
    def test_psi_zero_for_identical(self) -> None:
        x = np.linspace(0, 1, 10_000)
        psi = population_stability_index(x, x, bins=10)
        self.assertLess(psi, 1e-6)

    def test_psi_increases_when_shifted(self) -> None:
        baseline = np.random.default_rng(0).normal(0, 1, size=50_000)
        recent = baseline + 1.0
        psi = population_stability_index(baseline, recent, bins=10)
        self.assertGreater(psi, 0.1)


if __name__ == "__main__":
    unittest.main()

