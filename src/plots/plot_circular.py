import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import pygmsh

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '2d', 'Basic Shapes', 'Circular'))
from Circular_Boundary import Circular_Boundary


def build_disk_triangulation(radius: float, num_radial: int = 20, num_angular: int = 60):
    # Create concentric rings from r=0 to r=R
    radii = np.linspace(0.0, radius, num_radial)
    thetas = np.linspace(0.0, 2.0 * np.pi, num_angular, endpoint=False)

    points = []
    for ir, r in enumerate(radii):
        for it, th in enumerate(thetas if ir > 0 else [0.0]):  # single point at center
            points.append([r * np.cos(th), r * np.sin(th)])
    points = np.array(points)

    # Index helper for ring/angle to flat index
    def idx(ir, it):
        if ir == 0:
            return 0
        return 1 + (ir - 1) * num_angular + (it % num_angular)

    # Build triangle connectivity between rings
    tris = []
    for ir in range(1, num_radial):
        if ir == 1:
            # connect center to first ring
            for it in range(num_angular):
                c = 0
                a = idx(1, it)
                b = idx(1, it + 1)
                tris.append([c, a, b])
        else:
            # connect ring ir-1 to ring ir with quads split into two triangles
            for it in range(num_angular):
                a = idx(ir - 1, it)
                b = idx(ir - 1, it + 1)
                c = idx(ir, it)
                d = idx(ir, it + 1)
                tris.append([a, c, d])
                tris.append([a, d, b])

    tris = np.array(tris, dtype=int)
    triang = mtri.Triangulation(points[:, 0], points[:, 1], tris)
    return triang


def main():
    # Parameters (mirroring styles used in main.py)
    N = 1000
    dt = 0.01
    n_tot = 100
    e = 1.0
    mu = 1.0
    alpha = 1.0
    R = 5.0
    radius = R
    T_x0 = 1.0
    T_y0 = 1.0
    buckets_x = 16
    buckets_y = 16

    # Placeholder positions (will be re-initialized inside Circular_Boundary)
    positions = np.zeros((N, 2))

    # Run circular simulation
    positions, velocities, temperature_history = Circular_Boundary(
        N=N,
        R=R,
        positions=positions,
        radius=radius,
        T_x0=T_x0,
        T_y0=T_y0,
        dt=dt,
        n_tot=n_tot,
        e=e,
        mu=mu,
        alpha=alpha,
        buckets_x=buckets_x,
        buckets_y=buckets_y,
    )

    # Plot results similar to main.py
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Circular Domain DSMC (Specular Reflection)')

    # Final positions
    ax1 = axes[0, 0]
    ax1.scatter(positions[:, 0], positions[:, 1], s=1, alpha=0.6)
    ax1.set_aspect('equal', 'box')
    ax1.set_title('Final Particle Positions')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    circle = plt.Circle((0, 0), R, color='k', fill=False, linewidth=1.5)
    ax1.add_artist(circle)

    # Draw mesh like the provided example: pygmsh circle + grid overlay inside the circle
    CIRCLE_CENTER = [0.0, 0.0, 0.0]
    CIRCLE_RADIUS = R
    CIRCLE_MESH_SIZE = R / 12.0
    GRID_EXTENT = 1.0
    GRID_N_HORIZONTAL = 20
    GRID_N_VERTICAL = 20
    GRID_MESH_SIZE = R / 12.0
    MESH_LINE_WIDTH = 0.4
    GRID_LINE_WIDTH = 0.8
    GRID_ALPHA = 0.6
    CIRCLE_LINE_WIDTH = 1.5

    with pygmsh.geo.Geometry() as geom:
        circle = geom.add_circle(CIRCLE_CENTER, radius=CIRCLE_RADIUS, mesh_size=CIRCLE_MESH_SIZE)

        # Horizontal grid lines
        grid_y_values = np.linspace(-GRID_EXTENT * CIRCLE_RADIUS, GRID_EXTENT * CIRCLE_RADIUS, GRID_N_HORIZONTAL)
        for y in grid_y_values:
            if abs(y) < CIRCLE_RADIUS * 0.95:
                x_max = np.sqrt(CIRCLE_RADIUS**2 - y**2)
                p1 = geom.add_point([CIRCLE_CENTER[0] - x_max, CIRCLE_CENTER[1] + y, CIRCLE_CENTER[2]], mesh_size=GRID_MESH_SIZE)
                p2 = geom.add_point([CIRCLE_CENTER[0] + x_max, CIRCLE_CENTER[1] + y, CIRCLE_CENTER[2]], mesh_size=GRID_MESH_SIZE)
                geom.add_line(p1, p2)

        # Vertical grid lines
        grid_x_values = np.linspace(-GRID_EXTENT * CIRCLE_RADIUS, GRID_EXTENT * CIRCLE_RADIUS, GRID_N_VERTICAL)
        for xg in grid_x_values:
            if abs(xg) < CIRCLE_RADIUS * 0.95:
                y_max = np.sqrt(CIRCLE_RADIUS**2 - xg**2)
                p1 = geom.add_point([CIRCLE_CENTER[0] + xg, CIRCLE_CENTER[1] - y_max, CIRCLE_CENTER[2]], mesh_size=GRID_MESH_SIZE)
                p2 = geom.add_point([CIRCLE_CENTER[0] + xg, CIRCLE_CENTER[1] + y_max, CIRCLE_CENTER[2]], mesh_size=GRID_MESH_SIZE)
                geom.add_line(p1, p2)

        mesh = geom.generate_mesh()

    points = mesh.points[:, :2]
    cells = mesh.get_cells_type("triangle")

    # Plot triangular elements
    for tri in cells:
        poly = points[tri]
        ax1.fill(poly[:, 0], poly[:, 1], edgecolor="0.7", facecolor="none", linewidth=MESH_LINE_WIDTH)

    # Circle boundary
    theta = np.linspace(0.0, 2.0*np.pi, 200)
    circle_x = CIRCLE_CENTER[0] + CIRCLE_RADIUS * np.cos(theta)
    circle_y = CIRCLE_CENTER[1] + CIRCLE_RADIUS * np.sin(theta)
    ax1.plot(circle_x, circle_y, 'k-', linewidth=CIRCLE_LINE_WIDTH)

    # Overlay grid lines (re-draw for emphasis)
    for y in grid_y_values:
        if abs(y) < CIRCLE_RADIUS * 0.95:
            x_max = np.sqrt(CIRCLE_RADIUS**2 - y**2)
            ax1.plot([CIRCLE_CENTER[0] - x_max, CIRCLE_CENTER[0] + x_max],
                     [CIRCLE_CENTER[1] + y, CIRCLE_CENTER[1] + y],
                     'b-', linewidth=GRID_LINE_WIDTH, alpha=GRID_ALPHA)

    for xg in grid_x_values:
        if abs(xg) < CIRCLE_RADIUS * 0.95:
            y_max = np.sqrt(CIRCLE_RADIUS**2 - xg**2)
            ax1.plot([CIRCLE_CENTER[0] + xg, CIRCLE_CENTER[0] + xg],
                     [CIRCLE_CENTER[1] - y_max, CIRCLE_CENTER[1] + y_max],
                     'b-', linewidth=GRID_LINE_WIDTH, alpha=GRID_ALPHA)

    # Velocity scatter
    ax2 = axes[0, 1]
    ax2.scatter(velocities[:, 0], velocities[:, 1], s=1, alpha=0.6)
    ax2.set_title('Final Velocity Distribution')
    ax2.set_xlabel('vx')
    ax2.set_ylabel('vy')
    ax2.grid(True)

    # Temperature evolution
    ax3 = axes[1, 0]
    ax3.plot(np.arange(n_tot), temperature_history, linewidth=2)
    ax3.set_title('Temperature Evolution')
    ax3.set_xlabel('time step')
    ax3.set_ylabel('temperature')
    ax3.grid(True)

    # Speed histogram
    ax4 = axes[1, 1]
    speed = np.linalg.norm(velocities, axis=1)
    ax4.hist(speed, bins=50, density=True, alpha=0.7)
    ax4.set_title('Speed Distribution')
    ax4.set_xlabel('||v||')
    ax4.set_ylabel('pdf')
    ax4.grid(True)

    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'circular_simulation.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved circular simulation plot to: {output_path}")
    plt.close()


if __name__ == '__main__':
    main()


