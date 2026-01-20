import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the parent directory to the path to import from 2d/Arbitrary Shape
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '2d', 'Arbitrary Shape'))

from arbitrary_parameterized import Arbitrary_Shape_Parameterized
from arbitrary_helpers import sample_star_shape, create_mesh_from_star_shape

def plot_boundary(points, ax, color='black', linewidth=2, label='Boundary'):
    """
    Plot the boundary of the star shape.
    
    Parameters:
    points: Array of (x, y) coordinates defining the boundary
    ax: matplotlib axes object
    color: color of the boundary line
    linewidth: width of the boundary line
    label: label for the boundary
    """
    # Close the polygon by adding the first point at the end
    closed_points = np.vstack([points, points[0]])
    
    ax.plot(closed_points[:, 0], closed_points[:, 1], 
            color=color, linewidth=linewidth, label=label)

def plot_triangular_mesh(mesh, ax, color='gray', linewidth=0.5, alpha=0.3):
    """
    Plot the triangular mesh cells.
    
    Parameters:
    mesh: pygmsh mesh object
    ax: matplotlib axes object
    color: color of the mesh lines
    linewidth: width of the mesh lines
    alpha: transparency of the mesh lines
    """
    tri_conn = mesh.get_cells_type("triangle")
    pts2d = mesh.points[:, :2]
    
    # Plot each triangle
    for tri in tri_conn:
        if len(tri) >= 3:
            # Get the three vertices of the triangle
            p1 = pts2d[tri[0]]
            p2 = pts2d[tri[1]]
            p3 = pts2d[tri[2]]
            
            # Plot the three edges of the triangle
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 
                   color=color, linewidth=linewidth, alpha=alpha)
            ax.plot([p2[0], p3[0]], [p2[1], p3[1]], 
                   color=color, linewidth=linewidth, alpha=alpha)
            ax.plot([p3[0], p1[0]], [p3[1], p1[1]], 
                   color=color, linewidth=linewidth, alpha=alpha)

# Hard-coded Fourier coefficients for complex organic shape
# Using M=8 modes for highly detailed, intricate boundary
# Formula: r(t) = c0 + Σ[c_m*cos(m*t) + c_{m+M}*sin(m*t)] for m=1 to M
# Array structure: [c0, c1, c2, ..., c8, c9, c10, ..., c16]
#                  [base, ----cosine terms----, ----sine terms----]
# IMPORTANT: Keep sum of |coefficients| < c0 to avoid self-intersection
FOURIER_COEFFICIENTS = np.array([
    4.0,      # c0: base radius (must be > sum of absolute values of other coeffs)
    0.4,      # c1: 1-fold (asymmetry/shift)
    0.35,     # c2: 2-fold (elliptical component)
    0.45,     # c3: 3-fold (triangular component)
    0.20,     # c4: 4-fold (square-ish component)
    0.40,     # c5: 5-fold (pentagonal component)
    0.15,     # c6: 6-fold (hexagonal component)
    0.25,     # c7: 7-fold (heptagonal component)
    0.18,     # c8: 8-fold (octagonal component)
    0.25,     # c9: sine term for m=1 (rotation/phase)
    0.15,     # c10: sine term for m=2
    0.30,     # c11: sine term for m=3
    0.12,     # c12: sine term for m=4
    0.22,     # c13: sine term for m=5
    0.08,     # c14: sine term for m=6
    0.18,     # c15: sine term for m=7
    0.10      # c16: sine term for m=8
])  # Total: 2*8+1 = 17 coefficients for M=8 modes
# Sum of |coefficients| = 3.78, so c0=4.0 ensures r(t) > 0.2 always (no self-intersection)

# Hard-coded simulation parameters
SIMULATION_PARAMS = {
    'N': 3000,                    # Number of particles (increased for larger shape)
    'dt': 0.01,                   # Time step
    'n_tot': 150,                 # Total time steps
    'e': 1.0,                     # Energy parameter
    'mu': 1.0,                    # Mass parameter
    'alpha': 1.0,                 # Alpha parameter
    'T_x0': 1.0,                  # Initial x-temperature
    'T_y0': 1.0,                  # Initial y-temperature
    'num_boundary_points': 300    # Increased to capture complex boundary detail
}

def plot_velocity_field(positions, velocities, ax, scale=0.1):
    """
    Plot velocity vectors as arrows.
    
    Parameters:
    positions: array of particle positions
    velocities: array of particle velocities
    ax: matplotlib axes object
    scale: scaling factor for arrow length
    """
    if len(positions) == 0:
        return
    
    # Sample particles for visualization (too many arrows can be cluttered)
    if len(positions) > 500:
        indices = np.random.choice(len(positions), 500, replace=False)
        pos_sample = positions[indices]
        vel_sample = velocities[indices]
    else:
        pos_sample = positions
        vel_sample = velocities
    
    ax.quiver(pos_sample[:, 0], pos_sample[:, 1], 
              vel_sample[:, 0], vel_sample[:, 1],
              angles='xy', scale_units='xy', scale=1.0/scale, 
              alpha=0.6, width=0.003, color='red')

def main():
    print("=== Arbitrary Parameterized Shape 2D DSMC Simulation ===")
    print("This simulation uses Fourier coefficients to define complex organic boundaries.")
    
    # Use hard-coded Fourier coefficients
    fourier_coefficients = FOURIER_COEFFICIENTS
    M = (len(fourier_coefficients) - 1) // 2
    print(f"\nUsing {len(fourier_coefficients)} Fourier coefficients (M={M} modes)")
    print("Complex shape with multiple active harmonics (modes 1-8)")
    print(f"Fourier coefficients:")
    print(f"  Base radius (c0): {fourier_coefficients[0]:.2f}")
    print(f"  Cosine terms (c1-c{M}): {fourier_coefficients[1:M+1]}")
    print(f"  Sine terms (c{M+1}-c{2*M}): {fourier_coefficients[M+1:2*M+1]}")
    
    # Use hard-coded simulation parameters
    params = SIMULATION_PARAMS
    
    # Extract parameters
    N = params['N']
    dt = params['dt']
    n_tot = params['n_tot']
    e = params['e']
    mu = params['mu']
    alpha = params['alpha']
    T_x0 = params['T_x0']
    T_y0 = params['T_y0']
    num_boundary_points = params['num_boundary_points']
    
    print(f"\nSimulation parameters:")
    print(f"  Number of particles: {N}")
    print(f"  Time step: {dt}")
    print(f"  Total time steps: {n_tot}")
    print(f"  Energy parameter: {e}")
    print(f"  Mass parameter: {mu}")
    print(f"  Alpha parameter: {alpha}")
    print(f"  Initial temperatures: T_x0={T_x0}, T_y0={T_y0}")
    print(f"  Boundary points: {num_boundary_points}")
    
    # Use medium mesh_size for complex shape (need to capture details properly)
    # mesh_size values: 0.05 (very fine), 0.1 (default), 0.3 (coarse), 0.5 (very coarse)
    # Complex shapes need finer meshes to avoid meshing artifacts
    mesh_size = 0.25  # Medium mesh_size = moderate cell count, captures boundary details
    
    print(f"\nRunning simulation with {N} particles for {n_tot} time steps...")
    print(f"Boundary shape: {num_boundary_points} points sampling complex boundary")
    print(f"Active Fourier modes: {M} (creates highly detailed organic shape)")
    print(f"Mesh size: {mesh_size} (larger = fewer cells)")
    
    # Run the simulation (positions are generated internally with area weighting)
    positions, velocities, temperature_history, cell_list, boundary_points = Arbitrary_Shape_Parameterized(
        N=N,
        fourier_coefficients=fourier_coefficients,
        num_boundary_points=num_boundary_points,
        T_x0=T_x0,
        T_y0=T_y0,
        dt=dt,
        n_tot=n_tot,
        e=e,
        mu=mu,
        alpha=alpha,
        mesh_size=mesh_size,
    )
    
    # Recreate mesh for visualization with same mesh_size
    mesh = create_mesh_from_star_shape(boundary_points, mesh_size=mesh_size)
    
    print(f"Simulation completed!")
    print(f"Total mesh cells created: {len(cell_list)}")
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(18, 12))
    
    # Plot 1: Final particle positions with mesh
    ax1 = plt.subplot(2, 3, 1)
    if len(positions) > 0:
        ax1.scatter(positions[:, 0], positions[:, 1], s=3, alpha=0.6, c='blue', label='Particles')
    
    # Plot mesh - subtle lines to show cell structure
    plot_triangular_mesh(mesh, ax1, color='gray', linewidth=0.5, alpha=0.4)
    
    # Plot boundary
    plot_boundary(boundary_points, ax1, color='red', linewidth=2, label='Boundary')
    
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    ax1.set_title(f'Final Particle Positions (N={len(positions)}, Cells={len(cell_list)})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Plot 2: Velocity field
    ax2 = plt.subplot(2, 3, 2)
    if len(positions) > 0:
        # Color particles by speed
        speeds = np.linalg.norm(velocities, axis=1)
        scatter = ax2.scatter(positions[:, 0], positions[:, 1], 
                             c=speeds, s=2, alpha=0.6, cmap='viridis')
        plt.colorbar(scatter, ax=ax2, label='Speed')
        
        # Plot velocity vectors
        plot_velocity_field(positions, velocities, ax2, scale=0.1)
    
    plot_boundary(boundary_points, ax2, color='red', linewidth=2)
    ax2.set_xlabel('X Position')
    ax2.set_ylabel('Y Position')
    ax2.set_title('Velocity Field (colored by speed)')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    # Plot 3: Temperature history
    ax3 = plt.subplot(2, 3, 3)
    time_steps = np.arange(n_tot)
    ax3.plot(time_steps, temperature_history, 'r-', linewidth=2, label='Temperature')
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Temperature')
    ax3.set_title('Temperature Evolution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Velocity distribution (x-component)
    ax4 = plt.subplot(2, 3, 4)
    if len(velocities) > 0:
        ax4.hist(velocities[:, 0], bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax4.axvline(np.mean(velocities[:, 0]), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(velocities[:, 0]):.3f}')
        ax4.set_xlabel('Velocity X-component')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Velocity X Distribution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    # Plot 5: Velocity distribution (y-component)
    ax5 = plt.subplot(2, 3, 5)
    if len(velocities) > 0:
        ax5.hist(velocities[:, 1], bins=50, alpha=0.7, color='green', edgecolor='black')
        ax5.axvline(np.mean(velocities[:, 1]), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(velocities[:, 1]):.3f}')
        ax5.set_xlabel('Velocity Y-component')
        ax5.set_ylabel('Frequency')
        ax5.set_title('Velocity Y Distribution')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
    
    # Plot 6: Speed distribution
    ax6 = plt.subplot(2, 3, 6)
    if len(velocities) > 0:
        speeds = np.linalg.norm(velocities, axis=1)
        ax6.hist(speeds, bins=50, alpha=0.7, color='purple', edgecolor='black')
        ax6.axvline(np.mean(speeds), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(speeds):.3f}')
        ax6.set_xlabel('Speed')
        ax6.set_ylabel('Frequency')
        ax6.set_title('Speed Distribution')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print statistics
    print(f"\n=== Simulation Statistics ===")
    print(f"Final temperature: {temperature_history[-1]:.4f}")
    print(f"Average temperature: {np.mean(temperature_history):.4f}")
    print(f"Temperature std: {np.std(temperature_history):.4f}")
    print(f"Final particle count: {len(positions)}")
    
    if len(velocities) > 0:
        speeds = np.linalg.norm(velocities, axis=1)
        print(f"\nVelocity Statistics:")
        print(f"Mean speed: {np.mean(speeds):.4f}")
        print(f"Std speed: {np.std(speeds):.4f}")
        print(f"Mean vx: {np.mean(velocities[:, 0]):.4f}")
        print(f"Mean vy: {np.mean(velocities[:, 1]):.4f}")
    
    # Cell statistics
    particle_counts = [len(cell.particle_positions) for cell in cell_list]
    cell_areas = [cell.area() for cell in cell_list]
    
    print(f"\nMesh Statistics:")
    print(f"Mesh size parameter: {mesh_size}")
    print(f"Total cells: {len(cell_list)}")
    print(f"Particles per cell - Mean: {np.mean(particle_counts):.2f}, Std: {np.std(particle_counts):.2f}")
    print(f"Particles per cell - Min: {np.min(particle_counts)}, Max: {np.max(particle_counts)}")
    print(f"Cell areas - Mean: {np.mean(cell_areas):.4f}, Min: {np.min(cell_areas):.4f}, Max: {np.max(cell_areas):.4f}")
    print(f"Empty cells: {sum(1 for count in particle_counts if count == 0)}")

def create_animated_visualization():
    """
    Create an animated visualization of the simulation.
    Note: This requires modifying the simulation to store intermediate states.
    """
    print("\nAnimated visualization requires storing intermediate states.")
    print("This feature can be added by modifying the simulation function.")
    print("For now, please use the static visualization in main().")

if __name__ == "__main__":
    main()

