"""
Bayesian disease probability module for PROSPER.

Computes disease (class) probabilities from a distance matrix and reference
labels using exponential-weighted nearest neighbors.
"""

from .disease_probs import compute_disease_probs

__all__ = ["compute_disease_probs"]
