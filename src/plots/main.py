import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
import os

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '2d', 'Basic Shapes', 'Grid'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '3d', 'Box'))

from Grid_Boundary import Nanbu_Babovsky_2D_Periodic
from Box_Boundary import Nanbu_Babovsky_3D_Periodic

def run_2d_simulation():
    """Run 2D DSMC simulation"""
    print("="*60)
    print("RUNNING 2D NANBU-BABOVSKY DSMC SIMULATION")
    print("="*60)
    
    # Simulation parameters
    N = 1000          # Number of particles
    dt = 0.01         # Time step
    n_tot = 100       # Total number of time steps
    e = 1.0           # Energy parameter
    mu = 1.0          # Mass parameter
    alpha = 1.0       # Alpha parameter for VHS cross section
    Lx = 10.0         # Domain length in x
    Ly = 10.0         # Domain length in y
    ncx = 10          # Number of cells in x direction
    ncy = 10          # Number of cells in y direction
    S = 1.0           # Source term parameter
    dx = Lx / ncx     # Cell size in x direction
    T_x0 = 1.0        # Initial temperature in x direction
    T_y0 = 1.0        # Initial temperature in y direction
    
    print(f"Parameters: N={N}, dt={dt}, n_tot={n_tot}")
    print(f"Domain: Lx={Lx}, Ly={Ly}, cells: {ncx}x{ncy}")
    
    # Run the simulation with periodic boundary conditions
    positions, velocities, temperature_history = Nanbu_Babovsky_2D_Periodic(
        N, dt, n_tot, e, mu, alpha, Lx, Ly, ncx, ncy, S, dx, T_x0, T_y0, bc="pc"
    )
    
    print("2D Simulation completed!")
    
    # Create plots for 2D
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Nanbu-Babovsky 2D DSMC Simulation Results', fontsize=16)
    
    # Plot 1: Final particle positions
    ax1 = axes[0, 0]
    ax1.scatter(positions[:, 0], positions[:, 1], alpha=0.6, s=1)
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    ax1.set_title('Final Particle Positions')
    ax1.grid(True)
    ax1.set_xlim(0, Lx)
    ax1.set_ylim(0, Ly)
    
    # Plot 2: Final velocity distribution
    ax2 = axes[0, 1]
    ax2.scatter(velocities[:, 0], velocities[:, 1], alpha=0.6, s=1)
    ax2.set_xlabel('Vx')
    ax2.set_ylabel('Vy')
    ax2.set_title('Final Velocity Distribution')
    ax2.grid(True)
    
    # Plot 3: Temperature evolution
    ax3 = axes[1, 0]
    time_steps = np.arange(n_tot)
    ax3.plot(time_steps, temperature_history)
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Temperature')
    ax3.set_title('Temperature Evolution')
    ax3.grid(True)
    
    # Plot 4: Velocity magnitude distribution
    ax4 = axes[1, 1]
    velocity_magnitude = np.linalg.norm(velocities, axis=1)
    ax4.hist(velocity_magnitude, bins=50, alpha=0.7, density=True)
    ax4.set_xlabel('Velocity Magnitude')
    ax4.set_ylabel('Probability Density')
    ax4.set_title('Velocity Magnitude Distribution')
    ax4.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print some statistics
    print(f"\n2D Simulation Statistics:")
    print(f"Final average velocity magnitude: {np.mean(velocity_magnitude):.3f}")
    print(f"Final velocity std: {np.std(velocity_magnitude):.3f}")
    print(f"Final temperature: {temperature_history[-1]:.3f}")
    print(f"Temperature change: {temperature_history[-1] - temperature_history[0]:.3f}")

def run_3d_simulation():
    """Run 3D DSMC simulation"""
    print("\n" + "="*60)
    print("RUNNING 3D NANBU-BABOVSKY DSMC SIMULATION")
    print("="*60)
    
    # Simulation parameters
    N = 1000          # Number of particles
    dt = 0.01         # Time step
    n_tot = 100       # Total number of time steps
    e = 1.0           # Energy parameter
    mu = 1.0          # Mass parameter
    alpha = 1.0       # Alpha parameter for VHS cross section
    Lx = 10.0         # Domain length in x
    Ly = 10.0         # Domain length in y
    Lz = 10.0         # Domain length in z
    ncx = 8           # Number of cells in x direction (smaller for 3D)
    ncy = 8           # Number of cells in y direction
    ncz = 8           # Number of cells in z direction
    S = 1.0           # Source term parameter
    dx = Lx / ncx     # Cell size in x direction
    T_x0 = 1.0        # Initial temperature in x direction
    T_y0 = 1.0        # Initial temperature in y direction
    T_z0 = 1.0        # Initial temperature in z direction
    
    print(f"Parameters: N={N}, dt={dt}, n_tot={n_tot}")
    print(f"Domain: Lx={Lx}, Ly={Ly}, Lz={Lz}, cells: {ncx}x{ncy}x{ncz}")
    
    # Run the simulation with periodic boundary conditions
    positions, velocities, temperature_history = Nanbu_Babovsky_3D_Periodic(
        N, dt, n_tot, e, mu, alpha, Lx, Ly, Lz, ncx, ncy, ncz, S, dx, T_x0, T_y0, T_z0, bc="pc"
    )
    
    print("3D Simulation completed!")
    
    # Create plots for 3D
    fig = plt.figure(figsize=(15, 10))
    
    # Plot 1: Final particle positions (3D scatter)
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax1.scatter(positions[:, 0], positions[:, 1], positions[:, 2], alpha=0.6, s=1)
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    ax1.set_zlabel('Z Position')
    ax1.set_title('Final Particle Positions (3D)')
    ax1.set_xlim(0, Lx)
    ax1.set_ylim(0, Ly)
    ax1.set_zlim(0, Lz)
    
    # Plot 2: Final velocity distribution (3D scatter)
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    ax2.scatter(velocities[:, 0], velocities[:, 1], velocities[:, 2], alpha=0.6, s=1)
    ax2.set_xlabel('Vx')
    ax2.set_ylabel('Vy')
    ax2.set_zlabel('Vz')
    ax2.set_title('Final Velocity Distribution (3D)')
    
    # Plot 3: Temperature evolution
    ax3 = fig.add_subplot(2, 3, 3)
    time_steps = np.arange(n_tot)
    ax3.plot(time_steps, temperature_history)
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Temperature')
    ax3.set_title('Temperature Evolution')
    ax3.grid(True)
    
    # Plot 4: Velocity magnitude distribution
    ax4 = fig.add_subplot(2, 3, 4)
    velocity_magnitude = np.linalg.norm(velocities, axis=1)
    ax4.hist(velocity_magnitude, bins=50, alpha=0.7, density=True)
    ax4.set_xlabel('Velocity Magnitude')
    ax4.set_ylabel('Probability Density')
    ax4.set_title('Velocity Magnitude Distribution')
    ax4.grid(True)
    
    # Plot 5: Velocity components
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.hist(velocities[:, 0], bins=30, alpha=0.5, label='Vx', density=True)
    ax5.hist(velocities[:, 1], bins=30, alpha=0.5, label='Vy', density=True)
    ax5.hist(velocities[:, 2], bins=30, alpha=0.5, label='Vz', density=True)
    ax5.set_xlabel('Velocity')
    ax5.set_ylabel('Probability Density')
    ax5.set_title('Velocity Components Distribution')
    ax5.legend()
    ax5.grid(True)
    
    # Plot 6: Position distribution
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.hist(positions[:, 0], bins=30, alpha=0.5, label='X', density=True)
    ax6.hist(positions[:, 1], bins=30, alpha=0.5, label='Y', density=True)
    ax6.hist(positions[:, 2], bins=30, alpha=0.5, label='Z', density=True)
    ax6.set_xlabel('Position')
    ax6.set_ylabel('Probability Density')
    ax6.set_title('Position Distribution')
    ax6.legend()
    ax6.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print some statistics
    print(f"\n3D Simulation Statistics:")
    print(f"Final average velocity magnitude: {np.mean(velocity_magnitude):.3f}")
    print(f"Final velocity std: {np.std(velocity_magnitude):.3f}")
    print(f"Final temperature: {temperature_history[-1]:.3f}")
    print(f"Temperature change: {temperature_history[-1] - temperature_history[0]:.3f}")
    print(f"Average velocity components: Vx={np.mean(velocities[:,0]):.3f}, Vy={np.mean(velocities[:,1]):.3f}, Vz={np.mean(velocities[:,2]):.3f}")

def run_combined_simulation():
    """Run both 2D and 3D simulations simultaneously and display results together"""
    print("="*80)
    print("RUNNING COMBINED 2D & 3D NANBU-BABOVSKY DSMC SIMULATIONS")
    print("="*80)
    
    # Simulation parameters
    N = 1000          # Number of particles
    dt = 0.01         # Time step
    n_tot = 100       # Total number of time steps
    e = 1.0           # Energy parameter
    mu = 1.0          # Mass parameter
    alpha = 1.0       # Alpha parameter for VHS cross section
    Lx = 10.0         # Domain length in x
    Ly = 10.0         # Domain length in y
    Lz = 10.0         # Domain length in z (for 3D)
    ncx_2d = 10       # Number of cells in x direction (2D)
    ncy_2d = 10       # Number of cells in y direction (2D)
    ncx_3d = 8        # Number of cells in x direction (3D)
    ncy_3d = 8        # Number of cells in y direction (3D)
    ncz_3d = 8        # Number of cells in z direction (3D)
    S = 1.0           # Source term parameter
    dx_2d = Lx / ncx_2d     # Cell size in x direction (2D)
    dx_3d = Lx / ncx_3d     # Cell size in x direction (3D)
    T_x0 = 1.0        # Initial temperature in x direction
    T_y0 = 1.0        # Initial temperature in y direction
    T_z0 = 1.0        # Initial temperature in z direction
    
    print(f"Parameters: N={N}, dt={dt}, n_tot={n_tot}")
    print(f"2D Domain: Lx={Lx}, Ly={Ly}, cells: {ncx_2d}x{ncy_2d}")
    print(f"3D Domain: Lx={Lx}, Ly={Ly}, Lz={Lz}, cells: {ncx_3d}x{ncy_3d}x{ncz_3d}")
    
    # Run both simulations
    print("\nRunning 2D simulation...")
    pos_2d, vel_2d, temp_2d = Nanbu_Babovsky_2D_Periodic(
        N, dt, n_tot, e, mu, alpha, Lx, Ly, ncx_2d, ncy_2d, S, dx_2d, T_x0, T_y0, bc="pc"
    )
    print("2D Simulation completed!")
    
    print("\nRunning 3D simulation...")
    pos_3d, vel_3d, temp_3d = Nanbu_Babovsky_3D_Periodic(
        N, dt, n_tot, e, mu, alpha, Lx, Ly, Lz, ncx_3d, ncy_3d, ncz_3d, S, dx_3d, T_x0, T_y0, T_z0, bc="pc"
    )
    print("3D Simulation completed!")
    
    # Create simplified visualization with only temperature and positions
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Nanbu-Babovsky DSMC Simulation Results: Temperature & Positions', fontsize=16)
    
    time_steps = np.arange(n_tot)
    
    # Plot 1: 2D Temperature evolution
    ax1 = axes[0, 0]
    ax1.plot(time_steps, temp_2d, 'b-', linewidth=2, label='2D Simulation')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Temperature')
    ax1.set_title('2D Temperature Evolution')
    ax1.grid(True)
    ax1.legend()
    
    # Plot 2: 3D Temperature evolution
    ax2 = axes[0, 1]
    ax2.plot(time_steps, temp_3d, 'r-', linewidth=2, label='3D Simulation')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Temperature')
    ax2.set_title('3D Temperature Evolution')
    ax2.grid(True)
    ax2.legend()
    
    # Plot 3: 2D Final particle positions
    ax3 = axes[1, 0]
    ax3.scatter(pos_2d[:, 0], pos_2d[:, 1], alpha=0.6, s=1, c='blue')
    ax3.set_xlabel('X Position')
    ax3.set_ylabel('Y Position')
    ax3.set_title('2D: Final Particle Positions')
    ax3.grid(True)
    ax3.set_xlim(0, Lx)
    ax3.set_ylim(0, Ly)
    
    # Plot 4: 3D Final particle positions
    ax4 = axes[1, 1]
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    ax4.scatter(pos_3d[:, 0], pos_3d[:, 1], pos_3d[:, 2], alpha=0.6, s=1, c='red')
    ax4.set_xlabel('X Position')
    ax4.set_ylabel('Y Position')
    ax4.set_zlabel('Z Position')
    ax4.set_title('3D: Final Particle Positions')
    ax4.set_xlim(0, Lx)
    ax4.set_ylim(0, Ly)
    ax4.set_zlim(0, Lz)
    
    plt.tight_layout()
    plt.show()
    
    # Calculate velocity magnitudes for comparison
    vel_mag_2d = np.linalg.norm(vel_2d, axis=1)
    vel_mag_3d = np.linalg.norm(vel_3d, axis=1)
    
    # Print comparison statistics
    print(f"\n" + "="*80)
    print("SIMULATION COMPARISON STATISTICS")
    print("="*80)
    print(f"2D Simulation:")
    print(f"  Final average velocity magnitude: {np.mean(vel_mag_2d):.3f}")
    print(f"  Final velocity std: {np.std(vel_mag_2d):.3f}")
    print(f"  Final temperature: {temp_2d[-1]:.3f}")
    print(f"  Temperature change: {temp_2d[-1] - temp_2d[0]:.3f}")
    
    print(f"\n3D Simulation:")
    print(f"  Final average velocity magnitude: {np.mean(vel_mag_3d):.3f}")
    print(f"  Final velocity std: {np.std(vel_mag_3d):.3f}")
    print(f"  Final temperature: {temp_3d[-1]:.3f}")
    print(f"  Temperature change: {temp_3d[-1] - temp_3d[0]:.3f}")
    print(f"  Average velocity components: Vx={np.mean(vel_3d[:,0]):.3f}, Vy={np.mean(vel_3d[:,1]):.3f}, Vz={np.mean(vel_3d[:,2]):.3f}")
    
    print(f"\nComparison:")
    print(f"  Velocity magnitude ratio (3D/2D): {np.mean(vel_mag_3d)/np.mean(vel_mag_2d):.3f}")
    print(f"  Temperature ratio (3D/2D): {temp_3d[-1]/temp_2d[-1]:.3f}")

def run_boundary_condition_comparison():
    """Run simulations with different boundary conditions for comparison"""
    print("="*80)
    print("BOUNDARY CONDITION COMPARISON SIMULATION")
    print("="*80)
    
    # Simulation parameters
    N = 500          # Number of particles (smaller for faster comparison)
    dt = 0.01        # Time step
    n_tot = 50       # Total number of time steps (smaller for faster comparison)
    e = 1.0          # Energy parameter
    mu = 1.0         # Mass parameter
    alpha = 1.0      # Alpha parameter for VHS cross section
    Lx = 10.0        # Domain length in x
    Ly = 10.0        # Domain length in y
    Lz = 10.0        # Domain length in z
    ncx = 5          # Number of cells in x direction (smaller for faster comparison)
    ncy = 5          # Number of cells in y direction
    ncz = 5          # Number of cells in z direction
    S = 1.0          # Source term parameter
    dx = Lx / ncx    # Cell size in x direction
    T_x0 = 1.0       # Initial temperature in x direction
    T_y0 = 1.0       # Initial temperature in y direction
    T_z0 = 1.0       # Initial temperature in z direction
    
    boundary_conditions = ["pc", "rf", "mx"]
    bc_names = {"pc": "Periodic", "rf": "Reflecting", "mx": "Maxwell"}
    
    results = {}
    
    for bc in boundary_conditions:
        print(f"\nRunning 2D simulation with {bc_names[bc]} boundary condition...")
        pos_2d, vel_2d, temp_2d = Nanbu_Babovsky_2D_Periodic(
            N, dt, n_tot, e, mu, alpha, Lx, Ly, ncx, ncy, S, dx, T_x0, T_y0, bc=bc
        )
        
        print(f"Running 3D simulation with {bc_names[bc]} boundary condition...")
        pos_3d, vel_3d, temp_3d = Nanbu_Babovsky_3D_Periodic(
            N, dt, n_tot, e, mu, alpha, Lx, Ly, Lz, ncx, ncy, ncz, S, dx, T_x0, T_y0, T_z0, bc=bc
        )
        
        results[bc] = {
            '2d': {'pos': pos_2d, 'vel': vel_2d, 'temp': temp_2d},
            '3d': {'pos': pos_3d, 'vel': vel_3d, 'temp': temp_3d}
        }
    
    # Create temperature evolution plots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Temperature Evolution by Boundary Condition', fontsize=16)
    
    time_steps = np.arange(n_tot)
    
    # 2D temperature comparison
    ax1 = axes[0]
    for bc in boundary_conditions:
        ax1.plot(time_steps, results[bc]['2d']['temp'], label=f'2D {bc_names[bc]}', linewidth=2)
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Temperature')
    ax1.set_title('2D Temperature Evolution')
    ax1.legend()
    ax1.grid(True)
    
    # 3D temperature comparison
    ax2 = axes[1]
    for bc in boundary_conditions:
        ax2.plot(time_steps, results[bc]['3d']['temp'], label=f'3D {bc_names[bc]}', linewidth=2)
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Temperature')
    ax2.set_title('3D Temperature Evolution')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print comparison statistics
    print(f"\n" + "="*80)
    print("BOUNDARY CONDITION COMPARISON STATISTICS")
    print("="*80)
    for bc in boundary_conditions:
        print(f"\n{bc_names[bc]} Boundary Condition:")
        print(f"  2D Final Temperature: {results[bc]['2d']['temp'][-1]:.3f}")
        print(f"  3D Final Temperature: {results[bc]['3d']['temp'][-1]:.3f}")
        print(f"  2D Temperature Change: {results[bc]['2d']['temp'][-1] - results[bc]['2d']['temp'][0]:.3f}")
        print(f"  3D Temperature Change: {results[bc]['3d']['temp'][-1] - results[bc]['3d']['temp'][0]:.3f}")

def main():
    """Main function to run simulations"""
    import sys
    
    print("Nanbu-Babovsky DSMC Simulation Suite")
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        choice = sys.argv[1]
        if choice in ["-h", "--help"]:
            print("\nUsage: python main.py [option]")
            print("Options:")
            print("  1 or no argument: Run combined 2D and 3D simulations (default)")
            print("  2: Run boundary condition comparison")
            print("  -h, --help: Show this help message")
            return
    else:
        choice = "1"  # default
    
    if choice == "2":
        print("Running boundary condition comparison...")
        run_boundary_condition_comparison()
    else:
        print("Running combined 2D and 3D simulations...")
        run_combined_simulation()
    
    print("\nSimulation suite completed!")

if __name__ == "__main__":
    main()