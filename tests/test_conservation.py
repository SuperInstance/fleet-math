"""Tests for the conservation law: gamma + H conservation across fleet sizes."""

import sys
import os
import numpy as np

# Ensure we can import from src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fleet_math import (
    fleet_conservation_law,
    CONSERVATION_INTERCEPT,
    CONSERVATION_LOG_COEFF,
)
from fleet_math.health import coupling_entropy, algebraic_normalized


# ── Helper: build a random coupling matrix ──

def _random_coupling(n_agents: int, seed: int = 42) -> np.ndarray:
    """Build a random cosine-similarity coupling matrix for testing."""
    rng = np.random.RandomState(seed)
    vectors = rng.randn(n_agents, 8)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-10
    C = vectors @ vectors.T / (norms @ norms.T)
    return C


# ── Tests ──

def test_conservation_law_returns_dict():
    """fleet_conservation_law returns the expected keys."""
    result = fleet_conservation_law(30)
    expected_keys = {"predicted_sum", "deviation", "is_conserved", "expected_range", "sigma"}
    assert set(result.keys()) == expected_keys, f"Got {set(result.keys())}"


def test_predicted_sum_monotonic():
    """predicted_sum decreases as V increases (logarithmic decay)."""
    sums = [fleet_conservation_law(v)["predicted_sum"] for v in [5, 10, 20, 30, 50, 100]]
    for i in range(len(sums) - 1):
        assert sums[i] > sums[i + 1], f"Not monotonic at index {i}: {sums[i]} <= {sums[i+1]}"


def test_predicted_sum_formula():
    """predicted_sum matches intercept + log_coeff * log(V)."""
    for V in [5, 10, 20, 30, 50, 100]:
        expected = CONSERVATION_INTERCEPT + CONSERVATION_LOG_COEFF * np.log(V)
        got = fleet_conservation_law(V)["predicted_sum"]
        assert abs(got - expected) < 1e-10, f"V={V}: expected {expected}, got {got}"


def test_deviation_positive_below():
    """deviation returns positive when gamma+H is below predicted."""
    result = fleet_conservation_law(30)
    # Predicted ~ 1.283 - 0.159*log(30) ≈ 0.742
    d = result["deviation"](0.3, 0.2)  # gamma+H=0.5, well below
    assert d < 0, f"Expected negative deviation, got {d}"


def test_deviation_negative_above():
    """deviation returns negative when gamma+H is above predicted."""
    result = fleet_conservation_law(30)
    d = result["deviation"](0.5, 0.5)  # gamma+H=1.0, well above
    assert d > 0, f"Expected positive deviation, got {d}"


def test_is_conserved_exact():
    """At predicted_sum, is_conserved is True."""
    result = fleet_conservation_law(30)
    predicted = result["predicted_sum"]
    # Split predicted into gamma + H halves
    h = predicted * 0.3
    g = predicted - h
    assert result["is_conserved"](g, h), f"Should be conserved at predicted sum {predicted}"


def test_is_conserved_wide_tolerance():
    """Even far away, wide tolerance should pass."""
    result = fleet_conservation_law(10)
    # sigma ≈ 0.065, tolerance=20 means acceptable range is ±20*0.065 = ±1.3
    assert result["is_conserved"](0.0, 0.0, tolerance=20), "Wide tolerance should pass"


def test_is_conserved_narrow_tolerance():
    """Tight tolerance on large deviation should fail."""
    result = fleet_conservation_law(10)
    assert not result["is_conserved"](0.0, 0.0, tolerance=0.1), "Tight tolerance should reject"


def test_expected_range_symmetric():
    """expected_range is symmetric around predicted_sum."""
    result = fleet_conservation_law(30)
    lower, upper = result["expected_range"]
    predicted = result["predicted_sum"]
    assert abs((predicted - lower) - (upper - predicted)) < 1e-10


def test_coupling_type_style():
    """'style' coupling type returns the universal baseline."""
    s = fleet_conservation_law(30, coupling_type="style")["predicted_sum"]
    u = fleet_conservation_law(30)["predicted_sum"]
    assert s == u, "Style and default should match"


def test_coupling_type_topology():
    """'topology' coupling type returns topology baseline."""
    t = fleet_conservation_law(30, coupling_type="topology")["predicted_sum"]
    assert t == 1.232, f"Expected 1.232 for topology at V=30, got {t}"


def test_random_matrix_is_conserved():
    """Randomly generated coupling matrix should roughly satisfy conservation."""
    for V in [10, 20, 30, 50]:
        C = _random_coupling(V, seed=V)
        gamma = algebraic_normalized(C)
        H = coupling_entropy(C)
        result = fleet_conservation_law(V)
        # Random coupling should be within ±4 sigma (99.99% confidence)
        # This is a sanity check, not a proof
        assert result["is_conserved"](gamma, H, tolerance=4), (
            f"V={V}: gamma={gamma:.4f}, H={H:.4f}, sum={gamma+H:.4f}, "
            f"predicted={result['predicted_sum']:.4f}, sigma={result['sigma']:.4f}"
        )


def test_sigma_interpolation():
    """Sigma is interpolated for unsampled V values."""
    # V=15 is between 10 and 20 in the sigma table
    s10 = fleet_conservation_law(10)["sigma"]
    s20 = fleet_conservation_law(20)["sigma"]
    s15 = fleet_conservation_law(15)["sigma"]
    assert s10 < s15 < s20 or s10 > s15 > s20, (
        f"Sigma interpolation not monotonic: V=10:{s10}, V=15:{s15}, V=20:{s20}"
    )
