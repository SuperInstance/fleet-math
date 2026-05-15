"""
fleet-math v0.2.0 additions — Fleet health metrics
"""

import numpy as np

class _DummyCA:
    @staticmethod
    def laplacian(C):
        return np.diag(C.sum(axis=1)) - C
    @staticmethod
    def build_coupling(X):
        norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-10
        return X @ X.T / (norms @ norms.T)

ca = _DummyCA()

def coupling_entropy(C):
    e = np.linalg.eigvalsh(C)[::-1]
    p = np.abs(e) / (np.sum(np.abs(e)) + 1e-15)
    p = p[p > 1e-10]
    return float(-np.sum(p * np.log(p)) / np.log(len(e)))

def algebraic_normalized(C):
    L = ca.laplacian(C)
    e = np.linalg.eigvalsh(L)
    return float((e[1]-e[0]) / (e[-1]-e[0]+1e-15))

def timing_stability(timings):
    log_t = np.log(np.clip(timings, 1e-10, None))
    return float(1.0 / (1.0 + np.var(log_t)))

class FleetHealthMetric:
    _baseline_mu = None
    _baseline_sigma = None
    
    @classmethod
    def fit_baseline(cls, n_agents=30, n_samples=500, n_features=109):
        g, h, t = [], [], []
        for _ in range(n_samples):
            X = np.random.randn(n_agents, n_features)
            C = ca.build_coupling(X)
            g.append(algebraic_normalized(C))
            h.append(coupling_entropy(C))
            t.append(timing_stability(np.random.exponential(0.1, n_agents)))
        cls._baseline_mu = [np.mean(g), np.mean(h), np.mean(t)]
        cls._baseline_sigma = [np.std(g), np.std(h), np.std(t)]
    
    @classmethod
    def compute(cls, coupling_matrix, timings=None):
        if cls._baseline_mu is None:
            cls.fit_baseline()
        z = (algebraic_normalized(coupling_matrix) - cls._baseline_mu[0]) / cls._baseline_sigma[0]
        z += (coupling_entropy(coupling_matrix) - cls._baseline_mu[1]) / cls._baseline_sigma[1]
        z += ((timing_stability(timings) if timings is not None else 0.5) - cls._baseline_mu[2]) / cls._baseline_sigma[2]
        return z
    
    @classmethod
    def diagnose(cls, coupling_matrix, timings=None):
        z = cls.compute(coupling_matrix, timings)
        gamma = algebraic_normalized(coupling_matrix)
        H = coupling_entropy(coupling_matrix)
        if abs(z) < 1.0:
            return z, "healthy"
        if abs(z) < 2.0:
            clues = []
            if gamma < 0.05: clues.append("low_connectivity")
            if H < 0.9: clues.append("low_diversity")
            if gamma > 0.15 and H < 0.9: clues.append("consensus_herd")
            if gamma < 0.05 and H > 0.9: clues.append("chaotic_diverse")
            return z, "watch: " + ",".join(clues)
        return z, "anomaly: investigate"
