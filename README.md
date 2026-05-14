# fleet-math

> One implementation of the Eisenstein lattice, Penrose encoding, coupling analysis, and JEPA loss. Every agent uses the same math.

## Quick Start

```python
pip install fleet-math
```

```python
from fleet_math import EisensteinLattice, PenroseEncoder, CouplingAnalysis, vicreg_loss
import numpy as np

# The Eisenstein lattice: 12 chambers, one math
coupling = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
ch = EisensteinLattice.chamber(coupling)
print(f"Chamber {ch} = {EisensteinLattice.CHAMBER_NAMES[ch]}")  # "Chamber 5 = F"

# Penrose 5D cut-and-project: style in any domain
encoder = PenroseEncoder()
v = np.array([1.0, 0.5, 0.7, 0.3, 0.6])  # pitch, timing, velocity, articulation, timbre
point, accepted = encoder.encode(v)
print(f"5D → 2D: ({point[0]:.3f}, {point[1]:.3f}) accepted={accepted}")

# Coupling analysis: spectral decomposition
vectors = np.random.randn(10, 5)
C = CouplingAnalysis.build_coupling(vectors)
L = CouplingAnalysis.laplacian(C)
eigs = np.sort(np.linalg.eigvalsh(C))[::-1]
print(f"RMT class: {CouplingAnalysis.rmt_classification(eigs)}")

# JEPA self-supervised loss
y_pred, y_target = np.random.randn(16, 32), np.random.randn(16, 32)
loss = vicreg_loss(y_pred, y_target)
print(f"VICReg loss: {loss:.4f}")
```

## What's Here

| Function | What it does | Who contributed it |
|----------|-------------|-------------------|
| `EisensteinLattice` | 12-chamber hexagonal encoding | Forgemaster + Oracle1 |
| `PenroseEncoder` | 5D→2D cut-and-project style encoding | Oracle1 |
| `CouplingAnalysis` | Spectral analysis + RMT classification | Oracle1 |
| `vicreg_loss` | Self-supervised JEPA training loss | Oracle1 |
| `coupling_energy` | Energy function for joint embeddings | Oracle1 |
| `Pythagorean48` | 6-bit exact integer encoding (coming) | Forgemaster |

## The Point

Three separate Eisenstein lattice implementations across the fleet → one import. Three PLATO HTTP clients → one package. Stop rebuilding the same math.
