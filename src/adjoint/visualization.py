"""
Visualization utilities for DSMC shape optimisation.

Functions
---------
animate_simulation(history, C, ...)
    Animate particle positions over M time steps for a single run.

animate_comparison(hist_a, C_a, hist_b, C_b, ...)
    Side-by-side animation: initial boundary vs optimised boundary.

plot_convergence(obj_hist, grad_norm_hist, ax)
    Objective and gradient-norm convergence curves.

plot_objective_landscape_1d(C_opt, direction, sweep_range, sim_fn, ...)
    Sweep L along a 1-D line through C-space to verify the optimum is a minimum.

plot_final_density(hist_a, hist_b, box_half, inner_half, ...)
    Side-by-side 2-D particle-position histograms (heat maps).

Usage in Jupyter
----------------
    from adjoint.visualization import animate_comparison
    from IPython.display import HTML
    ani = animate_comparison(hist_init, C_init, hist_opt, C_opt, ...)
    HTML(ani.to_jshtml())          # inline interactive player
    # or:
    ani.save('comparison.gif', writer='pillow', fps=6)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from .boundary_geometry import gamma


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _boundary_pts(C, n: int = 300) -> np.ndarray:
    """Sample the Fourier boundary curve at n equi-spaced θ values."""
    thetas = np.linspace(0.0, 1.0, n)
    return np.array([gamma(t, C) for t in thetas])


def _draw_static(ax, C, box_half: float, inner_half: float,
                 color: str = 'royalblue', bdry_label: str = 'Boundary'):
    """Draw outer box, inner square, and Fourier boundary on ax."""
    ax.add_patch(plt.Rectangle(
        (-box_half, -box_half), 2 * box_half, 2 * box_half,
        fill=False, edgecolor='black', lw=2, label='Outer box'))
    ax.add_patch(plt.Rectangle(
        (-inner_half, -inner_half), 2 * inner_half, 2 * inner_half,
        facecolor='lightyellow', edgecolor='darkorange', lw=1.5,
        alpha=0.7, label='Inner square'))
    pts = _boundary_pts(C)
    ax.plot(pts[:, 0], pts[:, 1], color=color, lw=2, label=bdry_label)
    ax.set_xlim(-box_half * 1.12, box_half * 1.12)
    ax.set_ylim(-box_half * 1.12, box_half * 1.12)
    ax.set_aspect('equal')


def _traj_from_history(history):
    """
    Extract the particle-position trajectory as a list of (N,2) arrays.

    Returns M+1 snapshots: positions at the start of each step plus the
    final positions after the last step.
    """
    traj = [step.positions_start for step in history.steps]
    traj.append(history.final_positions)
    return traj


# ---------------------------------------------------------------------------
# Single-run animation
# ---------------------------------------------------------------------------

def animate_simulation(history, C, box_half: float, inner_half: float,
                       title: str = '', figsize=(6, 6),
                       interval: int = 200) -> FuncAnimation:
    """
    Animate particle positions over the recorded time steps.

    Parameters
    ----------
    history : SimulationHistory
    C : array-like
        Fourier boundary coefficients used for this run.
    box_half : float
        Half-width of the outer bounding square.
    inner_half : float
        Half-width of the inner observation square.
    title : str
    figsize : tuple
    interval : int
        Milliseconds between frames.

    Returns
    -------
    FuncAnimation
        Display in Jupyter with ``HTML(ani.to_jshtml())``.
    """
    traj = _traj_from_history(history)
    M    = len(traj) - 1

    fig, ax = plt.subplots(figsize=figsize)
    _draw_static(ax, C, box_half, inner_half)
    ax.set_title(title, fontsize=11)
    ax.legend(loc='upper right', fontsize=8)

    scat_out = ax.scatter([], [], s=14, c='steelblue', alpha=0.75, zorder=3)
    scat_in  = ax.scatter([], [], s=22, c='crimson',   alpha=0.95, zorder=4,
                          label='In inner sq')
    info = ax.text(0.02, 0.03, '', transform=ax.transAxes, va='bottom',
                   fontsize=9, bbox=dict(boxstyle='round,pad=0.3',
                                         facecolor='white', alpha=0.75))

    def _init():
        scat_out.set_offsets(np.empty((0, 2)))
        scat_in.set_offsets(np.empty((0, 2)))
        return scat_out, scat_in, info

    def _update(frame):
        x    = traj[frame]
        mask = (np.abs(x[:, 0]) <= inner_half) & (np.abs(x[:, 1]) <= inner_half)
        scat_out.set_offsets(x[~mask])
        scat_in.set_offsets(x[mask] if mask.any() else np.empty((0, 2)))
        info.set_text(f'Step {frame} / {M}   |   in inner sq: {mask.sum()}')
        return scat_out, scat_in, info

    ani = FuncAnimation(fig, _update, frames=M + 1, init_func=_init,
                        interval=interval, blit=True)
    plt.close(fig)
    return ani


# ---------------------------------------------------------------------------
# Side-by-side comparison animation
# ---------------------------------------------------------------------------

def animate_comparison(hist_a, C_a, hist_b, C_b,
                       box_half: float, inner_half: float,
                       label_a: str = 'Initial',
                       label_b: str = 'Optimised',
                       figsize=(13, 5.5),
                       interval: int = 200) -> FuncAnimation:
    """
    Side-by-side animation of two simulations run on different boundaries.

    Parameters
    ----------
    hist_a, hist_b : SimulationHistory
    C_a, C_b : array-like
    box_half, inner_half : float
    label_a, label_b : str
        Subplot titles.
    figsize, interval : standard matplotlib args.

    Returns
    -------
    FuncAnimation
    """
    traj_a = _traj_from_history(hist_a)
    traj_b = _traj_from_history(hist_b)
    M = min(len(traj_a), len(traj_b)) - 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    for ax, C, lbl in [(ax1, C_a, label_a), (ax2, C_b, label_b)]:
        _draw_static(ax, C, box_half, inner_half)
        ax.set_title(lbl, fontsize=12)
        ax.legend(loc='upper right', fontsize=7)

    # Scatter objects for both panels
    so_a = ax1.scatter([], [], s=14, c='steelblue', alpha=0.75, zorder=3)
    si_a = ax1.scatter([], [], s=22, c='crimson',   alpha=0.95, zorder=4)
    so_b = ax2.scatter([], [], s=14, c='steelblue', alpha=0.75, zorder=3)
    si_b = ax2.scatter([], [], s=22, c='crimson',   alpha=0.95, zorder=4)

    _kw = dict(va='bottom', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.75))
    txt_a = ax1.text(0.02, 0.03, '', transform=ax1.transAxes, **_kw)
    txt_b = ax2.text(0.02, 0.03, '', transform=ax2.transAxes, **_kw)

    step_lbl = fig.text(0.5, 0.97, '', ha='center', fontsize=12, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    def _init():
        for s in [so_a, si_a, so_b, si_b]:
            s.set_offsets(np.empty((0, 2)))
        return so_a, si_a, so_b, si_b, txt_a, txt_b, step_lbl

    def _update(frame):
        for so, si, txt, traj in [
                (so_a, si_a, txt_a, traj_a),
                (so_b, si_b, txt_b, traj_b)]:
            x    = traj[frame]
            mask = (np.abs(x[:, 0]) <= inner_half) & (np.abs(x[:, 1]) <= inner_half)
            so.set_offsets(x[~mask])
            si.set_offsets(x[mask] if mask.any() else np.empty((0, 2)))
            txt.set_text(f'particles in inner sq: {mask.sum()}')
        step_lbl.set_text(f'Step {frame} / {M}')
        return so_a, si_a, so_b, si_b, txt_a, txt_b, step_lbl

    ani = FuncAnimation(fig, _update, frames=M + 1, init_func=_init,
                        interval=interval, blit=True)
    plt.close(fig)
    return ani


# ---------------------------------------------------------------------------
# Convergence plot
# ---------------------------------------------------------------------------

def plot_convergence(obj_hist, grad_norm_hist=None, ax=None):
    """
    Plot objective value and (optionally) gradient norm vs iteration.

    Parameters
    ----------
    obj_hist : list of float
    grad_norm_hist : list of float or None
    ax : matplotlib Axes or None

    Returns
    -------
    ax
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    iters = np.arange(len(obj_hist))
    ax.plot(iters, obj_hist, 'b-', lw=1.8, label='Objective L')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('L', color='b')
    ax.grid(True, alpha=0.35)
    ax.set_title('Optimisation convergence')

    if grad_norm_hist is not None:
        ax2 = ax.twinx()
        ax2.semilogy(iters, grad_norm_hist, 'r--', lw=1.3, alpha=0.8,
                     label=r'$\|\nabla_C L\|$')
        ax2.set_ylabel(r'$\|\nabla_C L\|$', color='r')
        ax2.tick_params(axis='y', colors='r')
        lines1, lbl1 = ax.get_legend_handles_labels()
        lines2, lbl2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, lbl1 + lbl2, loc='upper right', fontsize=9)
    else:
        ax.legend()

    return ax


# ---------------------------------------------------------------------------
# 1-D objective landscape (proves the optimum is a local minimum)
# ---------------------------------------------------------------------------

def plot_objective_landscape_1d(C_opt, direction, sweep_range,
                                sim_fn, n_points: int = 25, ax=None):
    """
    Sweep L(C_opt + t·direction) over t ∈ sweep_range and plot vs t.

    A local minimum at t=0 confirms the optimised C is (at least locally)
    optimal in this direction.

    Parameters
    ----------
    C_opt : ndarray, shape (2N+1,)
        Optimised Fourier coefficients.
    direction : ndarray, shape (2N+1,)
        Direction in C-space to sweep along (need not be normalised).
    sweep_range : (float, float)
        (t_min, t_max).
    sim_fn : callable  C → float
        Evaluates L for a given C (wraps ForwardSimulation + objective).
    n_points : int
    ax : matplotlib Axes or None

    Returns
    -------
    t_vals : ndarray
    L_vals : ndarray
    ax
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    t_vals = np.linspace(sweep_range[0], sweep_range[1], n_points)
    L_vals = np.array([sim_fn(C_opt + t * direction) for t in t_vals])

    ax.plot(t_vals, L_vals, 'b-o', markersize=5, lw=1.5)
    ax.axvline(0.0, color='r', ls='--', lw=1.8, label='Optimised C  (t=0)')
    ax.set_xlabel('Perturbation  t  along sweep direction')
    ax.set_ylabel('Objective  L')
    ax.set_title('1-D objective landscape around optimum\n'
                 '(minimum at t=0 confirms local optimality)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.35)

    return t_vals, L_vals, ax


# ---------------------------------------------------------------------------
# 2-D particle density heat map
# ---------------------------------------------------------------------------

def plot_final_density(hist_a, hist_b, box_half: float, inner_half: float,
                       label_a: str = 'Initial',
                       label_b: str = 'Optimised',
                       n_bins: int = 30, figsize=(12, 5)):
    """
    Side-by-side 2-D histograms (heat maps) of final particle positions.

    Darker cells inside the inner square for the optimised run indicate
    more particles landing there (or closer to the origin, depending on
    the objective).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    bins = np.linspace(-box_half, box_half, n_bins + 1)

    vmax = None
    for ax, hist, lbl in [(ax1, hist_a, label_a), (ax2, hist_b, label_b)]:
        x = hist.final_positions
        H, xedges, yedges = np.histogram2d(x[:, 0], x[:, 1], bins=bins)
        if vmax is None:
            vmax = H.max() + 1
        im = ax.pcolormesh(xedges, yedges, H.T, cmap='YlOrRd',
                            vmin=0, vmax=vmax)
        ax.add_patch(plt.Rectangle(
            (-box_half, -box_half), 2*box_half, 2*box_half,
            fill=False, edgecolor='black', lw=2))
        ax.add_patch(plt.Rectangle(
            (-inner_half, -inner_half), 2*inner_half, 2*inner_half,
            fill=False, edgecolor='royalblue', lw=2, ls='--',
            label='Inner square'))
        ax.set_title(lbl, fontsize=12)
        ax.set_aspect('equal')
        ax.legend(fontsize=8)
        plt.colorbar(im, ax=ax, label='Particle count')

    fig.suptitle('Final particle density (darker = more particles)', fontsize=12)
    plt.tight_layout()
    return fig
