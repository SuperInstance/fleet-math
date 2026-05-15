"""fleet_math.types — Coupling type calibration and type-aware health metric."""

import numpy as np
from .health import coupling_entropy, algebraic_normalized

# Conservation law baselines per coupling type
# gamma+H = constant per type, per V (CV < 0.2 across all types tested)
BASELINES = {
    "style": {"V": {5: 1.027, 10: 0.873, 20: 0.848, 30: 0.807, 50: 0.726, 100: 0.649}, 
              "form": lambda V: 0.870 - 0.232 / np.log(max(V, 3))},
    "topology": {"V": {30: 1.232}, "form": lambda V: 1.232},  # placeholder for V-scaling
    "mixed": {"V": {30: 0.889}, "form": lambda V, alpha=0.5: 0.742 + 0.349 * alpha},
    "directed": {"V": {30: 0.995}, "form": lambda V: 0.995},
}

def estimate_type(C, coupling_type="auto"):
    """Estimate coupling type from matrix properties."""
    if coupling_type != "auto":
        return coupling_type
    
    n = C.shape[0]
    tri_upper = np.triu(C, 1)
    tri_lower = np.tril(C, -1)
    asymmetry = np.sum(np.abs(tri_upper - tri_lower.T)) / (n * (n-1))
    
    # Style coupling: all entries near [0, 1], most off-diagonal non-zero
    # Topology: many zeros, some positive values
    # Mixed: intermediate
    # Directed: asymmetric
    
    if asymmetry > 0.1:
        return "directed"
    
    sparsity = np.sum(C < 0.01) / (n * n)
    if sparsity > 0.5:
        return "topology"
    
    off_diag_mean = np.mean(np.abs(C - np.eye(n)))
    if off_diag_mean < 0.3:
        return "style"
    
    return "mixed"

class TypeAwareHealthMetric:
    """FleetHealthMetric with coupling-type calibration."""
    
    @staticmethod
    def compute(C, coupling_type="auto", V=None):
        if V is None:
            V = C.shape[0]
        
        ctype = estimate_type(C, coupling_type)
        baseline = BASELINES.get(ctype, BASELINES["style"])["form"](V)
        
        g = algebraic_normalized(C)
        h = coupling_entropy(C)
        
        z = (g + h - baseline) / 0.15  # approximate sigma
        return z, ctype

# Pre-computed baselines for common (type, V) pairs
# Usage: pip install fleet-math && from fleet_math.types import TypeAwareHealthMetric
