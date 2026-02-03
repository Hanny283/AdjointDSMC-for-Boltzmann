# Project Overview: Nanbu-Babovsky DSMC with Periodic Boundary Conditions and Shape Optimization

## Executive Summary

This project implements a comprehensive computational framework for optimizing star-shaped boundaries in particle-based kinetic simulations using the Direct Simulation Monte Carlo (DSMC) method. The system combines advanced physics simulation techniques with modern optimization algorithms to solve inverse design problems: given a target objective (minimizing kinetic energy in a specific region), determine the optimal boundary shape that achieves this goal. The framework supports both 2D and 3D geometries, with a particular focus on parameterized star-shaped boundaries defined by Fourier series expansions.

---

## Problem Statement and Motivation

The core challenge addressed by this project is the inverse design problem in kinetic gas dynamics: rather than simulating particle behavior in a given geometry, we seek to determine the optimal geometry that produces desired particle dynamics. Specifically, the project aims to minimize the kinetic energy (or "heat") of particles within a fixed inscribed square region by optimizing the shape of the surrounding boundary.

This problem has practical applications in engineering design, including:
- **Thermal management systems**: Designing container shapes that minimize heat concentration in critical regions
- **Plasma confinement**: Optimizing magnetic field boundaries to control particle energy distribution
- **Aerodynamic design**: Shaping boundaries to control flow characteristics in specific regions
- **Microfluidic devices**: Designing channel geometries for optimal particle transport

The mathematical formulation involves a constrained optimization problem where the objective function requires expensive physics simulations, making it computationally challenging and an ideal candidate for machine learning acceleration.

---

## Technical Architecture

### 1. Direct Simulation Monte Carlo (DSMC) Method

The project implements the Nanbu-Babovsky DSMC algorithm, a particle-based method for solving the Boltzmann equation that governs rarefied gas dynamics. The DSMC approach replaces the continuous distribution function with a finite set of computational particles, each representing a large number of real molecules.

**Key Components:**

**Particle Representation**: The system tracks thousands of particles (typically 3,000) in 2D space, each with position (x, y) and velocity (vx, vy) vectors. Particles are initialized with velocities sampled from a Maxwellian distribution at specified temperatures (T_x0, T_y0).

**Spatial Discretization**: The domain is discretized using triangular meshes generated via `pygmsh`, a Python interface to the Gmsh mesh generator. Each triangle cell maintains its own particle list, enabling efficient collision detection and boundary condition application. The mesh adapts to the boundary shape, with configurable mesh size parameters (typically 0.1-0.25) controlling the trade-off between accuracy and computational cost.

**Collision Algorithm**: The Nanbu-Babovsky method uses a probabilistic collision model within each cell. For each time step:
1. The expected number of collisions is computed based on particle density, relative velocities, and cross-sections
2. Particle pairs are randomly selected for potential collisions
3. Collisions are accepted or rejected based on acceptance-rejection sampling using relative velocity cross-sections
4. Accepted collisions update particle velocities using momentum and energy conservation

**Boundary Conditions**: The system implements reflecting boundary conditions for arbitrary shapes. When particles cross the boundary, their velocities are reflected specularly, preserving energy while changing direction. The boundary detection uses efficient geometric algorithms to determine when particles intersect the parameterized boundary curve.

**Particle Rebinning**: As particles move between cells, the system maintains a cell-particle invariant: each particle must reside in the cell containing its position. A sophisticated rebinning algorithm uses KD-trees for nearest-neighbor search and triangle-following algorithms to efficiently reassign particles to correct cells after each time step.

**Performance Optimizations**: The implementation includes several performance enhancements:
- Vectorized operations using NumPy for bulk particle updates
- KD-tree spatial indexing for O(log N) nearest-neighbor queries
- Efficient edge-to-cell adjacency mapping for boundary detection
- Parallel-ready architecture (though currently sequential)

### 2. Shape Parameterization

The project uses Fourier series to parameterize star-shaped boundaries, providing a compact and smooth representation suitable for optimization. A star-shaped domain is one where every point can be connected to a central point (the origin) by a straight line that lies entirely within the domain.

**Mathematical Formulation**: The boundary radius at angle θ is given by:
```
r(θ; c) = c₀ + Σ[m=1 to M] [c_m cos(mθ) + c_{m+M} sin(mθ)]
```

Where:
- `c₀` is the base radius (mean radius)
- `c_m` are cosine Fourier coefficients
- `c_{m+M}` are sine Fourier coefficients
- `M` is the number of Fourier modes (typically 4-8)

For M=4 modes, this yields 9 parameters (1 base + 4 cosine + 4 sine terms), providing sufficient flexibility to represent complex organic shapes while remaining computationally tractable.

**Geometric Properties**: The parameterization enables efficient computation of geometric properties:
- **Area**: Computed via numerical integration: Area = ½ ∫₀^{2π} r(θ)² dθ
- **Boundary points**: Sampled at regular angular intervals for mesh generation
- **Constraint checking**: Radial distance constraints can be evaluated at discrete angles

**Shape Validity**: The system enforces constraints to ensure the parameterization produces valid star-shaped domains:
- Positivity: r(θ) ≥ r_min > 0 for all θ (prevents self-intersection)
- Bounding box: r(θ) ≤ L / max(|cos θ|, |sin θ|) (keeps domain within [-L, L] × [-L, L])
- Square inscription: r(θ) ≥ a / max(|cos θ|, |sin θ|) (ensures square [-a, a] × [-a, a] fits inside)

### 3. Optimization Framework

The optimization module implements a complete workflow for shape optimization using evolutionary algorithms. The objective function minimizes kinetic energy in the target square region while satisfying geometric constraints.

**Objective Function**: For each candidate shape (defined by Fourier coefficients c):
1. Generate boundary points from Fourier coefficients
2. Create triangular mesh for the domain
3. Run DSMC simulation (typically 100 time steps for optimization, 150+ for final analysis)
4. Identify particles within the target square region [-a, a] × [-a, a]
5. Compute total kinetic energy: KE = Σ(vx² + vy²) for particles in square
6. Normalize by particle count: KE_per_particle = KE / N_in_square
7. Add spectral regularization: λ Σ m⁴(c_m² + c_{m+M}²) to penalize high-frequency oscillations
8. Return: Objective = KE_per_particle + Regularization

The regularization term encourages smooth boundaries by heavily penalizing high-frequency Fourier modes (weighted by m⁴), preventing jagged or oscillatory optimal shapes.

**Constraints**: The optimization enforces four constraint types:
1. **Square inscribed**: The target square must fit inside the domain (evaluated at 100 angular samples)
2. **Bounding box**: The domain must stay within a specified bounding box (prevents infinite growth)
3. **Area preservation**: The domain area must remain within 5% of the initial area (prevents trivial solutions)
4. **Positivity**: The radius must remain positive at all angles (ensures valid star-shape)

**Optimization Algorithm**: The system uses `scipy.optimize.differential_evolution`, a population-based global optimization algorithm that:
- Maintains a population of candidate solutions (typically 15 individuals)
- Uses mutation, crossover, and selection operations inspired by biological evolution
- Handles constraints through penalty methods or direct constraint satisfaction
- Supports parallel evaluation of objective functions (configurable workers)
- Includes local refinement ("polishing") at convergence

**Performance Characteristics**: Each objective evaluation requires:
- Mesh generation: ~0.1-1 seconds
- DSMC simulation: ~10-60 seconds (depending on particle count and time steps)
- Total: ~10-60 seconds per evaluation
- For 50 iterations with population size 15: ~750-4500 function evaluations = ~2-75 hours

This computational cost motivates the machine learning integration discussed in the ML integration plan.

### 4. Visualization and Tracking

The visualization module provides comprehensive tracking and real-time visualization of the optimization process.

**OptimizationTracker Class**: Maintains a complete history of:
- Iteration numbers and timestamps
- Fourier coefficients at each iteration
- Objective values and constraint violations
- Best solution found so far

**Visualization Features**: Creates multi-panel figures showing:
1. **Boundary and particles**: The current shape with particles colored by speed, overlaid with the target square
2. **Square region zoom**: Focused view of particles within the target square
3. **Objective history**: Evolution of objective value over iterations
4. **Coefficient evolution**: How Fourier coefficients change during optimization

**Output Management**: 
- Saves visualization frames at regular intervals (every 5 iterations)
- Generates progress plots showing objective and coefficient evolution
- Creates summary text files with optimization results
- Can compile frames into animations (requires imageio)

**Real-time Feedback**: The callback mechanism provides live updates during optimization, enabling users to monitor progress and detect issues early.

---

## Software Architecture

The project is organized into modular components:

**Core Simulation** (`src/2d/Arbitrary Shape/`):
- `arbitrary_parameterized.py`: Main DSMC simulation function
- `arbitrary_helpers.py`: Mesh generation, particle assignment, spatial queries
- `arbitrary_bc.py`: Boundary condition implementation

**Optimization** (`src/optimization/`):
- `config.py`: Centralized configuration management
- `shape_optimizer.py`: Objective function and geometric computations
- `constraints.py`: Constraint evaluation and builders
- `viz_optimizer.py`: Visualization and tracking
- `run_optimization.py`: Main optimization orchestration

**Supporting Infrastructure**:
- `cell_class.py`: Triangle cell implementation with particle management
- `edge_class.py`: Edge representation for mesh connectivity
- `universal_sim_helpers.py`: Shared utilities for collision cross-sections

**Dependencies**: The project uses minimal, well-established scientific Python libraries:
- `numpy`: Numerical computations
- `scipy`: Optimization algorithms
- `matplotlib`: Visualization
- `pygmsh`: Mesh generation

---

## Results and Applications

The framework successfully demonstrates:
1. **Feasibility**: Star-shaped boundaries can be optimized to minimize kinetic energy in target regions
2. **Constraint satisfaction**: All geometric constraints can be maintained during optimization
3. **Smooth solutions**: Regularization produces physically reasonable, smooth optimal shapes
4. **Scalability**: The modular architecture supports extension to 3D geometries and different objectives

**Typical Results**: Starting from a circle (radius 2.0), the optimizer explores the parameter space, typically finding shapes that:
- Maintain the inscribed square constraint
- Preserve area within tolerance
- Reduce kinetic energy in the target region through boundary shape modifications
- Exhibit smooth, non-oscillatory boundaries due to regularization

**Computational Performance**: 
- Single simulation: 10-60 seconds
- Full optimization (50 iterations): 2-75 hours (depending on parameters)
- Memory usage: ~100-500 MB (depends on particle count and mesh size)

---

## Future Directions and Machine Learning Integration

The project's computational cost makes it an ideal candidate for machine learning acceleration. The ML integration plan proposes:

**Surrogate Modeling**: Train neural networks to approximate the expensive DSMC simulation, mapping Fourier coefficients directly to kinetic energy. This could reduce evaluation time from seconds to milliseconds, enabling 100-1000x faster optimization.

**Active Learning**: Use uncertainty quantification to identify parameter regions where the surrogate is uncertain, then run expensive simulations only at those points to iteratively improve the model.

**Data Engineering**: Collect training data systematically across the parameter space, version control datasets with DVC, and maintain data quality through validation pipelines.

**Model Deployment**: Integrate trained surrogates into the optimization loop, with fallback to full simulations when uncertainty is high, ensuring both speed and accuracy.

This ML-enhanced version would transform the project from a computationally expensive research tool into a scalable, production-ready system suitable for real-world engineering applications.

---

## Conclusion

This project demonstrates a complete workflow for physics-informed shape optimization, combining advanced particle simulation methods with modern optimization techniques. The modular architecture, comprehensive visualization, and constraint handling make it a robust framework for inverse design problems in kinetic gas dynamics. The identified path for machine learning integration positions the project for significant performance improvements while maintaining physical accuracy, exemplifying the intersection of computational physics, optimization theory, and machine learning engineering.
