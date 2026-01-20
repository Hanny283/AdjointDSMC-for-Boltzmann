import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '3d', 'Spherical'))

from Spherical_Boundary import Spherical_Boundary


def main():
    # Parameters (mirroring styles used in plot_circular.py)
    N = 1000  # Reduced for 3D visualization performance
    dt = 0.01
    n_tot = 50  # Reduced for faster execution
    e = 1.0
    mu = 1.0
    alpha = 1.0
    R = 5.0
    radius = R
    T_x0 = 1.0
    T_y0 = 1.0
    T_z0 = 1.0
    buckets_x = 8
    buckets_y = 8
    buckets_z = 8

    # Placeholder positions (will be re-initialized inside Spherical_Boundary)
    positions = np.zeros((N, 3))

    # Run spherical simulation
    positions, velocities, temperature_history = Spherical_Boundary(
        N=N,
        R=R,
        positions=positions,
        radius=radius,
        T_x0=T_x0,
        T_y0=T_y0,
        T_z0=T_z0,
        dt=dt,
        n_tot=n_tot,
        e=e,
        mu=mu,
        alpha=alpha,
        buckets_x=buckets_x,
        buckets_y=buckets_y,
        buckets_z=buckets_z,
    )

    # Plot results similar to plot_circular.py but adapted for 3D
    fig = plt.figure(figsize=(15, 12))
    fig.suptitle('Spherical Domain DSMC (Specular Reflection)', fontsize=16)

    # 3D scatter plot of final positions
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax1.scatter(positions[:, 0], positions[:, 1], positions[:, 2], s=1, alpha=0.6)
    ax1.set_title('Final Particle Positions (3D)')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    
    # Draw sphere boundary
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    x_sphere = R * np.outer(np.cos(u), np.sin(v))
    y_sphere = R * np.outer(np.sin(u), np.sin(v))
    z_sphere = R * np.outer(np.ones(np.size(u)), np.cos(v))
    ax1.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.1, color='gray')

    # 2D projections
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.scatter(positions[:, 0], positions[:, 1], s=1, alpha=0.6)
    ax2.set_aspect('equal', 'box')
    ax2.set_title('Final Positions (XY projection)')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    circle = plt.Circle((0, 0), R, color='k', fill=False, linewidth=1.5)
    ax2.add_artist(circle)

    ax3 = fig.add_subplot(2, 3, 3)
    ax3.scatter(positions[:, 0], positions[:, 2], s=1, alpha=0.6)
    ax3.set_aspect('equal', 'box')
    ax3.set_title('Final Positions (XZ projection)')
    ax3.set_xlabel('x')
    ax3.set_ylabel('z')
    circle = plt.Circle((0, 0), R, color='k', fill=False, linewidth=1.5)
    ax3.add_artist(circle)

    # Velocity scatter plots
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.scatter(velocities[:, 0], velocities[:, 1], s=1, alpha=0.6)
    ax4.set_title('Final Velocity Distribution (vx vs vy)')
    ax4.set_xlabel('vx')
    ax4.set_ylabel('vy')
    ax4.grid(True)

    ax5 = fig.add_subplot(2, 3, 5)
    ax5.scatter(velocities[:, 0], velocities[:, 2], s=1, alpha=0.6)
    ax5.set_title('Final Velocity Distribution (vx vs vz)')
    ax5.set_xlabel('vx')
    ax5.set_ylabel('vz')
    ax5.grid(True)

    # Temperature evolution
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.plot(np.arange(n_tot), temperature_history, linewidth=2)
    ax6.set_title('Temperature Evolution')
    ax6.set_xlabel('time step')
    ax6.set_ylabel('temperature')
    ax6.grid(True)

    plt.tight_layout()
    output_path1 = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'spherical_simulation.png')
    plt.savefig(output_path1, dpi=300, bbox_inches='tight')
    print(f"Saved spherical simulation plot to: {output_path1}")
    plt.close()

    # Additional plots for speed distribution
    fig2, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig2.suptitle('Speed Analysis')

    # Speed histogram
    speed = np.linalg.norm(velocities, axis=1)
    axes[0].hist(speed, bins=50, density=True, alpha=0.7)
    axes[0].set_title('Speed Distribution')
    axes[0].set_xlabel('||v||')
    axes[0].set_ylabel('pdf')
    axes[0].grid(True)

    # Radial position histogram
    radial_positions = np.linalg.norm(positions, axis=1)
    axes[1].hist(radial_positions, bins=50, density=True, alpha=0.7)
    axes[1].set_title('Radial Position Distribution')
    axes[1].set_xlabel('r')
    axes[1].set_ylabel('pdf')
    axes[1].grid(True)

    plt.tight_layout()
    output_path2 = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'spherical_speed_analysis.png')
    plt.savefig(output_path2, dpi=300, bbox_inches='tight')
    print(f"Saved spherical speed analysis plot to: {output_path2}")
    plt.close()


if __name__ == '__main__':
    main()
