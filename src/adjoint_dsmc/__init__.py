"""
Adjoint DSMC package for arbitrary star-shaped boundary optimization.

Implements the discrete adjoint DSMC method from:
  - Caflisch, Silantyev, Yang (2021) — Maxwell molecules (Algorithm 3)
  - Yang, Silantyev, Caflisch (2023) — General collision kernels (Algorithm 2)

Extended to the spatially inhomogeneous case by adjointing through the
specular reflecting boundary condition.

Public API
----------
forward_pass_with_history  -- run forward DSMC and record full history
backward_adjoint_pass      -- initialize adjoint and run backward sweep
compute_gradient_wrt_fourier -- accumulate ∂J/∂c from BC reflection events
check_gradient_fd          -- finite-difference gradient check utility
VHSKernel                  -- Variable Hard Sphere collision kernel
MaxwellKernel              -- Maxwell molecules (constant kernel)
ForwardHistory             -- dataclass holding all forward-pass records
"""

from .records import CollisionStepRecord, BCStepRecord, ForwardHistory
from .kernels import CollisionKernel, MaxwellKernel, VHSKernel
from .forward_wrapper import forward_pass_with_history
from .backward import initialize_adjoint, backward_adjoint_pass
from .boundary import adjoint_through_reflection, boundary_normal_gradient
from .gradient import compute_gradient_wrt_fourier, check_gradient_fd

__all__ = [
    "CollisionStepRecord",
    "BCStepRecord",
    "ForwardHistory",
    "CollisionKernel",
    "MaxwellKernel",
    "VHSKernel",
    "forward_pass_with_history",
    "initialize_adjoint",
    "backward_adjoint_pass",
    "adjoint_through_reflection",
    "boundary_normal_gradient",
    "compute_gradient_wrt_fourier",
    "check_gradient_fd",
]
