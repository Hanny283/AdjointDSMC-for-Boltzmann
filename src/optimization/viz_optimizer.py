"""
Visualization and tracking module for optimization progress.

This module provides tools for tracking optimization iterations and creating
real-time visualizations of the optimization process.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
from datetime import datetime


class OptimizationTracker:
    """
    Track optimization progress and maintain history.
    
    Stores iteration data including coefficients, objective values, and
    constraint violations. Provides methods for plotting and analysis.
    """
    
    def __init__(self, output_dir='optimization_results'):
        """
        Initialize the tracker.
        
        Parameters
        ----------
        output_dir : str, optional
            Directory to save output files (default: 'optimization_results')
        """
        self.history = []
        self.best_c = None
        self.best_value = np.inf
        self.iteration_count = 0
        self.output_dir = output_dir
        self.start_time = datetime.now()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Create subdirectories
        self.frames_dir = os.path.join(output_dir, 'frames')
        os.makedirs(self.frames_dir, exist_ok=True)
        
    def update(self, c, obj_val, constraint_values=None):
        """
        Update tracker with new iteration data.
        
        Parameters
        ----------
        c : array-like
            Current Fourier coefficients
        obj_val : float
            Current objective value
        constraint_values : dict, optional
            Dictionary of constraint values
        """
        self.iteration_count += 1
        
        # Store history
        entry = {
            'iteration': self.iteration_count,
            'c': np.array(c).copy(),
            'objective': obj_val,
            'constraints': constraint_values.copy() if constraint_values else None,
            'timestamp': datetime.now()
        }
        self.history.append(entry)
        
        # Update best solution
        if obj_val < self.best_value:
            self.best_value = obj_val
            self.best_c = np.array(c).copy()
            print(f"\n  *** New best: {self.best_value:.6f} at iteration {self.iteration_count}")
        
    def get_history_arrays(self):
        """
        Extract history as numpy arrays for plotting.
        
        Returns
        -------
        iterations : ndarray
            Iteration numbers
        objectives : ndarray
            Objective values
        coefficients : ndarray
            Coefficient matrix (iterations × num_params)
        """
        if not self.history:
            return np.array([]), np.array([]), np.array([])
        
        iterations = np.array([h['iteration'] for h in self.history])
        objectives = np.array([h['objective'] for h in self.history])
        coefficients = np.array([h['c'] for h in self.history])
        
        return iterations, objectives, coefficients
    
    def plot_progress(self, save=True):
        """
        Plot optimization progress.
        
        Creates a figure with:
        1. Objective value vs iteration
        2. Coefficient evolution
        
        Parameters
        ----------
        save : bool, optional
            Save the figure (default: True)
        """
        if not self.history:
            print("No history to plot yet.")
            return
        
        iterations, objectives, coefficients = self.get_history_arrays()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Objective evolution
        ax1 = axes[0]
        ax1.plot(iterations, objectives, 'b-', linewidth=2, label='Objective')
        ax1.axhline(self.best_value, color='r', linestyle='--', 
                    linewidth=1, label=f'Best: {self.best_value:.4f}')
        ax1.set_xlabel('Iteration', fontsize=12)
        ax1.set_ylabel('Objective Value', fontsize=12)
        ax1.set_title('Optimization Progress', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Coefficient evolution
        ax2 = axes[1]
        M = (coefficients.shape[1] - 1) // 2
        
        # Plot c0 separately (different scale)
        ax2_twin = ax2.twinx()
        ax2_twin.plot(iterations, coefficients[:, 0], 'k-', linewidth=2.5, 
                     label='c₀ (base)', alpha=0.7)
        ax2_twin.set_ylabel('c₀ (base radius)', fontsize=11, color='k')
        ax2_twin.tick_params(axis='y', labelcolor='k')
        
        # Plot Fourier coefficients
        for m in range(1, M + 1):
            ax2.plot(iterations, coefficients[:, m], '-', 
                    linewidth=1.5, alpha=0.7, label=f'c{m} (cos)')
            ax2.plot(iterations, coefficients[:, m + M], '--', 
                    linewidth=1.5, alpha=0.7, label=f'c{m+M} (sin)')
        
        ax2.set_xlabel('Iteration', fontsize=12)
        ax2.set_ylabel('Fourier Coefficients', fontsize=11)
        ax2.set_title('Coefficient Evolution', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper left', fontsize=8, ncol=2)
        ax2_twin.legend(loc='upper right', fontsize=8)
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(self.output_dir, 'optimization_progress.png')
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"  Saved progress plot: {filepath}")
        
        plt.close()
    
    def save_summary(self):
        """Save optimization summary to text file."""
        filepath = os.path.join(self.output_dir, 'optimization_summary.txt')
        
        with open(filepath, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("OPTIMIZATION SUMMARY\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"Start time: {self.start_time}\n")
            f.write(f"End time: {datetime.now()}\n")
            f.write(f"Total iterations: {self.iteration_count}\n\n")
            
            f.write(f"Best objective value: {self.best_value:.8f}\n")
            f.write(f"Best coefficients:\n")
            for i, c in enumerate(self.best_c):
                f.write(f"  c[{i}] = {c:12.8f}\n")
            
            f.write("\n" + "=" * 70 + "\n")
        
        print(f"  Saved summary: {filepath}")


def visualize_iteration(c, results, opt_config, iteration, tracker=None):
    """
    Create comprehensive visualization for a single optimization iteration.
    
    Creates a 4-panel figure showing:
    1. Boundary shape with particles (colored by speed)
    2. Particles in target square (highlighted)
    3. Objective history (if tracker provided)
    4. Fourier coefficients bar chart
    
    Parameters
    ----------
    c : array-like
        Current Fourier coefficients
    results : dict
        Results from evaluate_with_details
    opt_config : dict
        Optimization configuration
    iteration : int
        Current iteration number
    tracker : OptimizationTracker, optional
        Tracker for history plotting
    """
    if results is None:
        print(f"  WARNING: Cannot visualize iteration {iteration} - results is None")
        return
    
    fig = plt.figure(figsize=(16, 12))
    
    # Extract data
    positions = results['positions']
    velocities = results['velocities']
    boundary_points = results['boundary_points']
    a = opt_config['a']
    
    # Compute speeds
    speeds = np.linalg.norm(velocities, axis=1) if len(velocities) > 0 else np.array([])
    
    # Identify particles in square
    in_square_mask = (np.abs(positions[:, 0]) <= a) & (np.abs(positions[:, 1]) <= a)
    
    # === Panel 1: Boundary + All Particles ===
    ax1 = plt.subplot(2, 2, 1)
    
    # Plot particles colored by speed
    if len(positions) > 0:
        scatter1 = ax1.scatter(positions[:, 0], positions[:, 1], 
                              c=speeds, s=3, alpha=0.6, cmap='viridis',
                              vmin=0, vmax=np.percentile(speeds, 95) if len(speeds) > 0 else 1)
        plt.colorbar(scatter1, ax=ax1, label='Speed', fraction=0.046, pad=0.04)
    
    # Plot boundary
    closed_boundary = np.vstack([boundary_points, boundary_points[0]])
    ax1.plot(closed_boundary[:, 0], closed_boundary[:, 1], 
            'r-', linewidth=2.5, label='Boundary')
    
    # Plot target square
    square = Rectangle((-a, -a), 2*a, 2*a, 
                      fill=False, edgecolor='blue', linewidth=2, 
                      linestyle='--', label='Target Square')
    ax1.add_patch(square)
    
    ax1.set_xlabel('x', fontsize=11)
    ax1.set_ylabel('y', fontsize=11)
    ax1.set_title(f'Iteration {iteration}: Boundary and Particles', 
                 fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # === Panel 2: Particles in Square (Zoomed) ===
    ax2 = plt.subplot(2, 2, 2)
    
    # Plot all particles (faded)
    if len(positions) > 0:
        ax2.scatter(positions[:, 0], positions[:, 1], 
                   c='gray', s=2, alpha=0.2, label='All particles')
    
    # Highlight particles in square
    if np.any(in_square_mask):
        pos_in_square = positions[in_square_mask]
        vel_in_square = velocities[in_square_mask]
        speeds_in_square = np.linalg.norm(vel_in_square, axis=1)
        
        scatter2 = ax2.scatter(pos_in_square[:, 0], pos_in_square[:, 1],
                              c=speeds_in_square, s=15, alpha=0.8, 
                              cmap='hot', edgecolors='black', linewidth=0.5,
                              label='In square')
        plt.colorbar(scatter2, ax=ax2, label='Speed (in square)', 
                    fraction=0.046, pad=0.04)
    
    # Plot square boundary
    square2 = Rectangle((-a, -a), 2*a, 2*a,
                       fill=False, edgecolor='blue', linewidth=2.5)
    ax2.add_patch(square2)
    
    ax2.set_xlim(-a*1.5, a*1.5)
    ax2.set_ylim(-a*1.5, a*1.5)
    ax2.set_xlabel('x', fontsize=11)
    ax2.set_ylabel('y', fontsize=11)
    ax2.set_title(f'Target Square Region (N={results["num_particles_in_square"]}/{results["num_particles_total"]})',
                 fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    # === Panel 3: Objective History ===
    ax3 = plt.subplot(2, 2, 3)
    
    if tracker and len(tracker.history) > 0:
        iterations, objectives, _ = tracker.get_history_arrays()
        ax3.plot(iterations, objectives, 'b-', linewidth=2, marker='o', markersize=4)
        ax3.axhline(tracker.best_value, color='r', linestyle='--', 
                   linewidth=1.5, label=f'Best: {tracker.best_value:.4f}')
        ax3.scatter([iteration], [results['objective']], 
                   c='red', s=100, marker='*', zorder=5, label='Current')
        ax3.set_xlabel('Iteration', fontsize=11)
        ax3.set_ylabel('Objective Value', fontsize=11)
        ax3.set_title('Objective Evolution', fontsize=13, fontweight='bold')
        ax3.legend(fontsize=9)
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'Objective History\n(Not available yet)', 
                ha='center', va='center', fontsize=12, transform=ax3.transAxes)
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
    
    # === Panel 4: Fourier Coefficients ===
    ax4 = plt.subplot(2, 2, 4)
    
    M = (len(c) - 1) // 2
    indices = np.arange(len(c))
    
    # Color code: base (c0), cosine terms, sine terms
    colors = ['black']  # c0
    colors += ['blue'] * M  # cosine terms
    colors += ['red'] * M   # sine terms
    
    bars = ax4.bar(indices, c, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    
    # Labels
    labels = ['c₀']
    labels += [f'c{m}' for m in range(1, M+1)]
    labels += [f'c{m+M}' for m in range(1, M+1)]
    
    ax4.set_xticks(indices)
    ax4.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax4.set_xlabel('Coefficient', fontsize=11)
    ax4.set_ylabel('Value', fontsize=11)
    ax4.set_title('Fourier Coefficients', fontsize=13, fontweight='bold')
    ax4.axhline(0, color='gray', linestyle='-', linewidth=0.5)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='black', alpha=0.7, label='Base (c₀)'),
        Patch(facecolor='blue', alpha=0.7, label='Cosine terms'),
        Patch(facecolor='red', alpha=0.7, label='Sine terms')
    ]
    ax4.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    # Add text box with metrics
    textstr = f'Objective: {results["objective"]:.4f}\n'
    textstr += f'KE/particle: {results["ke_per_particle"]:.4f}\n'
    textstr += f'Regularization: {results["regularization"]:.4f}\n'
    textstr += f'Area: {results["area"]:.2f}'
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    fig.text(0.02, 0.98, textstr, fontsize=10, verticalalignment='top',
            bbox=props, family='monospace')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save figure
    if tracker:
        filepath = os.path.join(tracker.frames_dir, f'iter_{iteration:04d}.png')
        plt.savefig(filepath, dpi=120, bbox_inches='tight')
        print(f"  Saved visualization: {filepath}")
    
    plt.close()


def create_optimization_animation(tracker, output_filename='optimization_animation.mp4'):
    """
    Create animation from saved frames.
    
    Note: Requires ffmpeg or imageio for MP4 creation.
    
    Parameters
    ----------
    tracker : OptimizationTracker
        Tracker containing frame directory
    output_filename : str, optional
        Output filename (default: 'optimization_animation.mp4')
    """
    try:
        import imageio
    except ImportError:
        print("WARNING: imageio not installed. Cannot create animation.")
        print("Install with: pip install imageio imageio-ffmpeg")
        return
    
    frames_dir = tracker.frames_dir
    output_path = os.path.join(tracker.output_dir, output_filename)
    
    # Get all frame files
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])
    
    if len(frame_files) == 0:
        print("No frames found to create animation.")
        return
    
    print(f"Creating animation from {len(frame_files)} frames...")
    
    # Read frames
    images = []
    for filename in frame_files:
        filepath = os.path.join(frames_dir, filename)
        images.append(imageio.imread(filepath))
    
    # Save as MP4
    imageio.mimsave(output_path, images, fps=2, codec='libx264')
    
    print(f"Animation saved: {output_path}")


if __name__ == "__main__":
    # Test the tracker
    print("Testing OptimizationTracker...")
    
    tracker = OptimizationTracker(output_dir='test_optimization_results')
    
    # Simulate some iterations
    for i in range(10):
        c = np.random.randn(9) * 0.1
        c[0] = 2.0  # base radius
        obj = np.random.rand() + i * 0.1
        
        tracker.update(c, obj)
    
    # Plot progress
    tracker.plot_progress(save=True)
    tracker.save_summary()
    
    print("\nTracker test completed!")
    print(f"  Total iterations: {tracker.iteration_count}")
    print(f"  Best value: {tracker.best_value:.6f}")
    print(f"  Output directory: {tracker.output_dir}")
