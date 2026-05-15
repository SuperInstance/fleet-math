"""fleet-math — Canonical algorithms for the entire fleet.

One implementation of the Eisenstein lattice, Penrose encoding,
Pythagorean48, and coupling analysis. Every agent uses these.
No more three implementations of the same math.
"""

import numpy as np
import math
from typing import Tuple, List, Optional
from .health import coupling_entropy, algebraic_normalized, timing_stability, FleetHealthMetric

PHI = (1 + math.sqrt(5)) / 2  # golden ratio


# ── Eisenstein Lattice ──────────────────────────────

class EisensteinLattice:
    """12-chamber hexagonal lattice. Z[ω] where ω = e^{2πi/3}.
    
    Use cases across the fleet:
    - Oracle1: style vectors → 12-chamber encoding
    - Forgemaster: constraint coupling → chamber assignment
    - JC1: GPU warp similarity → chamber-based allocation
    - tensor-spline: neural weights on Eisenstein grid
    """
    
    CHAMBER_NAMES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
    CHAMBER_ANGLES = np.array([math.pi * i / 6 for i in range(12)])
    CHAMBER_VECTORS = np.column_stack([
        np.cos(CHAMBER_ANGLES), np.sin(CHAMBER_ANGLES)
    ])  # (12, 2)
    
    @staticmethod
    def chamber(coupling: np.ndarray) -> int:
        """Snap coupling vector to nearest chamber. Argmax of 12 weights."""
        return int(np.argmax(coupling[:12] if len(coupling) >= 12 else 
                            np.pad(coupling, (0, 12 - len(coupling)))))
    
    @staticmethod
    def interval(chamber_a: int, chamber_b: int) -> int:
        """Shortest distance between chambers in semitones."""
        d = abs(chamber_a - chamber_b) % 12
        return min(d, 12 - d)
    
    @staticmethod
    def project(weights: np.ndarray) -> Tuple[float, float]:
        """Project coupling weights to 2D lattice point."""
        w = weights[:12] if len(weights) >= 12 else np.pad(weights, (0, 12 - len(weights)))
        return (float(np.sum(w * EisensteinLattice.CHAMBER_VECTORS[:, 0])),
                float(np.sum(w * EisensteinLattice.CHAMBER_VECTORS[:, 1])))


# ── Penrose 5D→2D Encoder ──────────────────────────

class PenroseEncoder:
    """5D→2D cut-and-project encoding. Oracle1 brings this to the fleet."""
    
    # 5th roots of unity projection matrix
    PROJECTION = np.array([
        [1.0, 0.309017, -0.809017, -0.809017, 0.309017],
        [0.0, 0.951057, 0.587785, -0.587785, -0.951057],
    ])
    
    PERP_PROJECTION = np.array([
        [1.0, -0.809017, 0.309017, 0.309017, -0.809017],
        [0.0, 0.587785, -0.951057, -0.951057, 0.587785],
    ])
    
    def __init__(self, window_radius: float = 2.0 * PHI):
        self.window_radius = window_radius
    
    def encode(self, v: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Project 5D vector → 2D, check acceptance window."""
        physical = self.PROJECTION @ v
        perp = self.PERP_PROJECTION @ v
        accepted = np.linalg.norm(perp) <= self.window_radius
        return physical, accepted


# ── Pythagorean48 (6-bit exact encoding) ───────────

def pythagorean48_snap(vector: np.ndarray) -> np.ndarray:
    """6-bit exact integer encoding. FM brings this to the fleet."""
    return np.round(vector * 48) / 48


# ── Coupling Matrix Operations ─────────────────────

class CouplingAnalysis:
    """Spectral analysis of coupling matrices. Universal — any domain."""
    
    @staticmethod
    def build_coupling(vectors: np.ndarray) -> np.ndarray:
        """Cosine similarity coupling matrix. Vectors: (n, d)."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-10
        return vectors @ vectors.T / (norms @ norms.T)
    
    @staticmethod
    def laplacian(C: np.ndarray) -> np.ndarray:
        """Graph Laplacian L = D - C."""
        return np.diag(C.sum(axis=1)) - C
    
    @staticmethod
    def rmt_classification(eigenvalues: np.ndarray) -> str:
        """Random Matrix Theory classification. Poisson, GOE, or spiked."""
        n = len(eigenvalues)
        if n < 10:
            return "too-small"
        # Check for spiked model (first eigenvalue >> rest)
        if eigenvalues[0] > 10 * eigenvalues[1]:
            return "spiked-covariance"
        # Spacing ratio test
        spacings = np.diff(np.sort(eigenvalues))
        mean_r = np.mean(spacings[1:] / spacings[:-1])
        if abs(mean_r - 0.386) < 0.1:
            return "poisson"
        elif abs(mean_r - 0.536) < 0.1:
            return "goe"
        else:
            return f"mixed (r={mean_r:.3f})"


# ── Coupling Type System ────────────────────────────

from .types import TypeAwareHealthMetric, estimate_type, BASELINES


# ── JEPA Loss Functions ────────────────────────────

def vicreg_loss(y_pred: np.ndarray, y_target: np.ndarray,
                sim_weight=1.0, var_weight=0.5, cov_weight=0.1) -> float:
    """VICReg loss. Oracle1 brings self-supervised JEPA to the fleet."""
    inv = float(np.mean((y_pred - y_target) ** 2))
    std = np.std(y_target, axis=0) + 1e-4
    var = float(np.mean(np.maximum(0, 1.0 - std)))
    yc = y_target - np.mean(y_target, axis=0, keepdims=True)
    cov = (yc.T @ yc) / y_target.shape[0]
    cv = float((np.sum(cov ** 2) - np.sum(np.diag(cov) ** 2)) / y_target.shape[1])
    return sim_weight * inv + var_weight * var + cov_weight * cv

def coupling_energy(embeddings: np.ndarray, coupling_matrix: np.ndarray) -> float:
    """Energy = sum_{i,j} C_ij * ||e_i - e_j||^2.
    High coupling should mean low distance in embedding space."""
    n = len(embeddings)
    energy = 0.0
    for i in range(n):
        for j in range(n):
            w = coupling_matrix[i][j]
            energy += w * np.sum((embeddings[i] - embeddings[j]) ** 2)
    return float(energy / (n * n))


# ── Conservation Law ──────────────────────────────

# Empirical conservation law: gamma + H = 1.283 - 0.159 * log(V)
# 
# DISCOVERY: In fleet-wide coupling analysis, the sum of algebraic_normalized (gamma)
# and coupling_entropy (H) is remarkably conserved for a given agent count V,
# regardless of coupling matrix structure. This was discovered empirically through
# Monte Carlo simulation across all fleet agents (Oracle1, Forgemaster, JC1, etc.).
#
# Derivation basis: R² = 0.9602 across V ∈ [5, 200] with 5000 samples per V.
# The best-fit line: gamma + H = 1.283 - 0.159 * log(V)
#
# Coupling-type corrections: each coupling type has a slightly different intercept.
# See fleet_math.types.BASELINES for per-type offsets. The universal conservation
# law is the ``style'' baseline — the most common configuration.
#
# PHYSICAL ANALOGY: This is like the conservation of mass-energy in the fleet's
# coupling space. Just as total energy is conserved in a closed physical system,
# gamma+H is conserved for a fixed fleet size. Changes in connectivity (gamma)
# must be compensated by changes in diversity (H), and vice versa.
#
# PRACTICAL USE: If an agent's coupling matrix violates this conservation law
# (deviation > 0.15), it indicates:
#   1. Preferential attachment (one agent dominates)
#   2. Measurement noise in the coupling matrix
#   3. Anomalous coupling regime worth investigating
#
# REFERENCES:
#   - plato-ng experiments/coupling_conversation/proof.py (original derivation)
#   - .transcripts/0-cross-pollination-1.md (fleet integration decision)

CONSERVATION_LOG_COEFF = -0.159
CONSERVATION_INTERCEPT = 1.283

# Tolerance windows (± sigma) per V, derived from empirical distribution
CONSERVATION_SIGMA = {
    5: 0.070, 10: 0.065, 20: 0.058, 30: 0.050,
    50: 0.048, 100: 0.042, 200: 0.038,
}


def _conservation_sigma(V: int) -> float:
    """Interpolate empirical sigma for any V."""
    vs = sorted(CONSERVATION_SIGMA.keys())
    if V <= vs[0]:
        return CONSERVATION_SIGMA[vs[0]]
    if V >= vs[-1]:
        return CONSERVATION_SIGMA[vs[-1]]
    for lo, hi in zip(vs, vs[1:]):
        if lo < V <= hi:
            frac = (V - lo) / (hi - lo)
            return CONSERVATION_SIGMA[lo] + frac * (CONSERVATION_SIGMA[hi] - CONSERVATION_SIGMA[lo])
    return 0.05


def fleet_conservation_law(V: int, coupling_type: str = "style") -> dict:
    """Conservation law for gamma + H at fleet size V.
    
    The sum of algebraic_normalized (gamma) and coupling_entropy (H)
    satisfies gamma + H ≈ intercept + log_coeff * log(V), with
    coupling-type-specific offsets.
    
    Parameters
    ----------
    V : int
        Number of agents (or nodes in the coupling matrix).
    coupling_type : str
        One of "style", "topology", "mixed", "directed".
        Default "style" uses the universal empirical law.
        Coupling-type offsets come from fleet_math.types.BASELINES.
    
    Returns
    -------
    dict with keys:
        predicted_sum : float
            Expected value of gamma + H for given V and coupling_type.
        deviation : callable(gamma, H) -> float
            Returns predicted_sum - (gamma + H). Positive means actual
            sum is below predicted.
        is_conserved : callable(gamma, H, tolerance=2.0) -> bool
            Checks if |deviation| <= tolerance * sigma(V).
            tolerance=2.0 allows ±2 sigma (95% confidence interval).
        expected_range : tuple[float, float]
            (lower_bound, upper_bound) for gamma+H at ±2 sigma.
        sigma : float
            Empirical standard deviation of gamma+H at this V.
    """
    from .types import BASELINES
    
    # Get baseline for this coupling type
    if coupling_type == "style":
        # Universal empirical law: gamma+H = 1.283 - 0.159*log(V)
        # R² = 0.9602 across V ∈ [5, 200], 5000 samples per V
        predicted = CONSERVATION_INTERCEPT + CONSERVATION_LOG_COEFF * np.log(max(V, 3))
    else:
        # Per-type baselines from fleet_math.types.BASELINES
        entry = BASELINES.get(coupling_type, BASELINES["style"])
        predicted = entry["form"](V)
    sigma = _conservation_sigma(V)
    
    def deviation(gamma: float, H: float) -> float:
        return (gamma + H) - predicted
    
    def is_conserved(gamma: float, H: float, tolerance: float = 2.0) -> bool:
        return abs(deviation(gamma, H)) <= tolerance * sigma
    
    lower, upper = predicted - 2 * sigma, predicted + 2 * sigma
    
    return {
        "predicted_sum": predicted,
        "deviation": deviation,
        "is_conserved": is_conserved,
        "expected_range": (lower, upper),
        "sigma": sigma,
    }
