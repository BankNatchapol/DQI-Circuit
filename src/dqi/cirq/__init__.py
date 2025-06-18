"""
Cirq implementation of Decoded Quantum Interferometry (DQI).

This module provides Cirq-specific implementations of DQI circuits,
including state preparation, Dicke state preparation, and decoding algorithms.
"""

from .initialization import UnaryAmplitudeEncoding, get_optimal_w
from .dicke_state_preparation import UnkGate
from .decoding import GJEGate, gauss_jordan_operations_general

__all__ = [
    'UnaryAmplitudeEncoding',
    'get_optimal_w',
    'UnkGate', 
    'GJEGate',
    'gauss_jordan_operations_general'
] 