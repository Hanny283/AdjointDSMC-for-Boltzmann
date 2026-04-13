"""
Optimization module for star-shaped boundary optimization.

This module provides tools for optimizing Fourier coefficients of star-shaped
boundaries to minimize particle kinetic energy in a specified region.
"""

from .config import OPTIMIZATION_CONFIG, SIMULATION_CONFIG
from .shape_optimizer import (
    objective_function,
    compute_radius,
    compute_area,
    particles_in_square,
    compute_kinetic_energy
)
from .constraints import build_constraints
from .viz_optimizer import OptimizationTracker, visualize_iteration

__all__ = [
    'OPTIMIZATION_CONFIG',
    'SIMULATION_CONFIG',
    'objective_function',
    'compute_radius',
    'compute_area',
    'particles_in_square',
    'compute_kinetic_energy',
    'build_constraints',
    'OptimizationTracker',
    'visualize_iteration'
]
