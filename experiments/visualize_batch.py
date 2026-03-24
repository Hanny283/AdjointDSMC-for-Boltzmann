"""
Visualise all boundaries found across a batch run and report coefficient statistics.

Produces:
  <output_dir>/all_boundaries.png      — all 30 shapes overlaid + colourbar
  <output_dir>/boundary_grid.png       — 6×5 grid of individual shapes
  <output_dir>/coefficient_stats.png   — mean±std bar chart + box plots
  <output_dir>/coefficient_table.txt   — per-coefficient statistics table

Usage (from the repo root):
  python3 experiments/visualize_batch.py
  python3 experiments/visualize_batch.py --results-dir batch_results
"""

import sys
import os
import json
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Polygon as MplPolygon

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
for p in [_root, os.path.join(_root, 'src')]:
    if p not in sys.path:
        sys.path.insert(0, p)


# ---------------------------------------------------------------------------
# Fourier helpers
# ---------------------------------------------------------------------------

def _shape_xy(c, n=400):
    M = (len(c) - 1) // 2
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = np.full_like(theta, c[0])
    for m in range(1, M + 1):
        r += c[m] * np.cos(m * theta) + c[M + m] * np.sin(m * theta)
    return r * np.cos(theta), r * np.sin(theta), theta


def _coef_labels(M):
    return ['c₀'] + [f'c{m}·cos' for m in range(1, M + 1)] + [f'c{m}·sin' for m in range(1, M + 1)]


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_runs(results_dir):
    path = os.path.join(results_dir, 'batch_summary.json')
    if not os.path.exists(path):
        raise FileNotFoundError(f"No batch_summary.json in '{results_dir}'")
    with open(path) as f:
        summary = json.load(f)
    runs = [r for r in summary['runs']
            if 'objective' in r and r['objective'] < 1e9]
    if not runs:
        raise ValueError("No valid runs found in batch_summary.json")
    return summary, runs


# ---------------------------------------------------------------------------
# Figure 1 — all boundaries overlaid
# ---------------------------------------------------------------------------

def plot_all_boundaries(runs, a, L, output_path):
    objs = np.array([r['objective'] for r in runs])
    coeffs = [np.array(r['coefficients']) for r in runs]
    n = len(runs)

    # Colour by objective rank (best = most vivid)
    norm = Normalize(vmin=objs.min(), vmax=objs.max())
    cmap = plt.cm.plasma_r

    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor('#0f1117')
    ax.set_facecolor('#1a1d27')

    for spine in ax.spines.values():
        spine.set_edgecolor('#3a3d4a')
    ax.tick_params(colors='#b0b0b0')
    ax.xaxis.label.set_color('#c0c0c0')
    ax.yaxis.label.set_color('#c0c0c0')

    # Sort so best shapes are drawn on top
    order = np.argsort(objs)[::-1]

    for rank, idx in enumerate(order):
        c = coeffs[idx]
        x, y, _ = _shape_xy(c)
        lw  = 2.5 if rank == n - 1 else 0.9   # best on top = thickest
        alp = 1.0 if rank == n - 1 else 0.55
        ax.plot(x, y, lw=lw, alpha=alp,
                color=cmap(norm(objs[idx])),
                zorder=rank)

    # Inscribed square
    sq = MplPolygon([[-a, -a], [a, -a], [a, a], [-a, a]],
                    closed=True, fill=False, edgecolor='#ff6b6b',
                    linewidth=2.0, linestyle='--', zorder=n + 1)
    ax.add_patch(sq)

    # Bounding box (faint)
    ax.plot([-L, L, L, -L, -L], [-L, -L, L, L, -L],
            lw=0.5, color='#555', ls=':', zorder=0)

    # Colourbar
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label('Objective value', color='#c0c0c0', fontsize=10)
    cb.ax.tick_params(labelcolor='#b0b0b0', labelsize=8)

    lim = max(2.5, np.abs(np.vstack([_shape_xy(c)[:2] for c in coeffs])).max() + 0.3)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    ax.set_xlabel('x', fontsize=11)
    ax.set_ylabel('y', fontsize=11)
    ax.set_title(
        f'All {n} converged boundaries  |  '
        f'obj ∈ [{objs.min():.4f}, {objs.max():.4f}]',
        color='#f0f0f0', fontsize=13, fontweight='bold', pad=12
    )

    fig.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved -> {output_path}")


# ---------------------------------------------------------------------------
# Figure 2 — individual grid
# ---------------------------------------------------------------------------

def plot_boundary_grid(runs, a, output_path, ncols=5):
    objs  = np.array([r['objective'] for r in runs])
    coeffs = [np.array(r['coefficients']) for r in runs]
    n = len(runs)

    # Sort best → worst
    order = np.argsort(objs)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
    fig.patch.set_facecolor('#0f1117')
    axes_flat = axes.flatten()

    for rank, idx in enumerate(order):
        ax = axes_flat[rank]
        ax.set_facecolor('#1a1d27')
        for sp in ax.spines.values():
            sp.set_edgecolor('#3a3d4a')
        ax.tick_params(colors='#808080', labelsize=6)

        c = coeffs[idx]
        x, y, _ = _shape_xy(c)
        ax.plot(x, y, lw=1.4, color='#4a9eff')

        sq = MplPolygon([[-a, -a], [a, -a], [a, a], [-a, a]],
                        closed=True, fill=False, edgecolor='#ff6b6b',
                        linewidth=1.2, linestyle='--')
        ax.add_patch(sq)

        lim = max(abs(x).max(), abs(y).max()) * 1.1
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect('equal')

        r = runs[idx]
        ax.set_title(
            f'#{rank+1}  seed={r["seed"]}\nobj={objs[idx]:.4f}',
            color='#d0d0d0', fontsize=7, pad=3
        )
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide unused subplots
    for ax in axes_flat[n:]:
        ax.set_visible(False)

    fig.suptitle('Individual boundaries (sorted best → worst)',
                 color='#f0f0f0', fontsize=13, fontweight='bold', y=1.01)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved -> {output_path}")


# ---------------------------------------------------------------------------
# Figure 3 — coefficient statistics
# ---------------------------------------------------------------------------

def plot_coefficient_stats(runs, output_path):
    objs  = np.array([r['objective'] for r in runs])
    coeffs = np.array([r['coefficients'] for r in runs])   # (n, n_coeff)
    n, n_coeff = coeffs.shape
    M = (n_coeff - 1) // 2
    labels = _coef_labels(M)

    mean_c = coeffs.mean(axis=0)
    std_c  = coeffs.std(axis=0)
    min_c  = coeffs.min(axis=0)
    max_c  = coeffs.max(axis=0)

    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor('#0f1117')
    gs = gridspec.GridSpec(2, 2, figure=fig,
                           hspace=0.45, wspace=0.35,
                           left=0.08, right=0.97, top=0.92, bottom=0.08)

    ax_face  = '#1a1d27'
    spine_c  = '#3a3d4a'
    text_kw  = dict(color='#e0e0e0')
    title_kw = dict(color='#f0f0f0', fontsize=11, fontweight='bold', pad=8)

    def _style(ax):
        ax.set_facecolor(ax_face)
        for sp in ax.spines.values():
            sp.set_edgecolor(spine_c)
        ax.tick_params(colors='#b0b0b0', labelsize=8)
        ax.xaxis.label.set_color('#c0c0c0')
        ax.yaxis.label.set_color('#c0c0c0')

    x = np.arange(n_coeff)

    # ── Panel 1: mean ± std bar chart ────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    _style(ax1)
    palette = ['#ff9f43'] + ['#4a9eff'] * M + ['#f38ba8'] * M
    bars = ax1.bar(x, mean_c, yerr=std_c, capsize=5,
                   color=palette, alpha=0.78, edgecolor='#0f1117',
                   linewidth=0.5, ecolor='#ffd166', error_kw=dict(lw=1.5))

    # Plot each individual run as a scatter overlay
    jitter = np.random.default_rng(0).uniform(-0.25, 0.25, (n, n_coeff))
    for i in range(n):
        ax1.scatter(x + jitter[i], coeffs[i], s=8, alpha=0.35,
                    color='white', zorder=5)

    # Mark the best-run coefficients
    best_idx = int(np.argmin(objs))
    ax1.plot(x, coeffs[best_idx], 'r*', markersize=11,
             label=f'Best run (seed={runs[best_idx]["seed"]})', zorder=10)

    ax1.axhline(0, color='#3a3d4a', lw=1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_ylabel('Value', **text_kw)
    ax1.set_title('Fourier coefficients — mean ± std  (dots = all runs)', **title_kw)
    ax1.legend(fontsize=8, facecolor=ax_face, edgecolor=spine_c,
               labelcolor='#c0c0c0')

    # ── Panel 2: coefficient std dev (how much each mode varies) ─────────
    ax2 = fig.add_subplot(gs[1, 0])
    _style(ax2)
    ax2.bar(x, std_c, color=palette, alpha=0.78, edgecolor='#0f1117', linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=7.5, rotation=45, ha='right')
    ax2.set_ylabel('Std dev', **text_kw)
    ax2.set_title('Coefficient variability (σ)', **title_kw)

    # ── Panel 3: range [min, max] per coefficient ─────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    _style(ax3)
    ax3.bar(x, max_c - min_c, bottom=min_c,
            color=palette, alpha=0.60, edgecolor='#0f1117', linewidth=0.5)
    ax3.plot(x, mean_c, 'o-', color='#ffd166', markersize=4, lw=1.2, label='mean')
    ax3.axhline(0, color='#3a3d4a', lw=1)
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, fontsize=7.5, rotation=45, ha='right')
    ax3.set_ylabel('Value', **text_kw)
    ax3.set_title('Coefficient range [min, max]', **title_kw)
    ax3.legend(fontsize=8, facecolor=ax_face, edgecolor=spine_c,
               labelcolor='#c0c0c0')

    fig.suptitle(
        f'Coefficient Statistics — {n} runs  |  '
        f'obj mean={objs.mean():.4f}  std={objs.std():.4f}',
        color='#f0f0f0', fontsize=13, fontweight='bold', y=0.97
    )

    fig.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved -> {output_path}")


# ---------------------------------------------------------------------------
# Text table
# ---------------------------------------------------------------------------

def write_coefficient_table(runs, output_path):
    objs   = np.array([r['objective'] for r in runs])
    coeffs = np.array([r['coefficients'] for r in runs])
    n, n_coeff = coeffs.shape
    M = (n_coeff - 1) // 2
    labels = _coef_labels(M)

    mean_c = coeffs.mean(axis=0)
    std_c  = coeffs.std(axis=0)
    min_c  = coeffs.min(axis=0)
    max_c  = coeffs.max(axis=0)

    best_idx = int(np.argmin(objs))
    best_c   = coeffs[best_idx]

    lines = [
        "FOURIER COEFFICIENT STATISTICS",
        "=" * 75,
        f"Runs analysed : {n}",
        f"Objective     : min={objs.min():.6f}  mean={objs.mean():.6f}  "
        f"std={objs.std():.6f}  max={objs.max():.6f}",
        "",
        f"{'Mode':<10}{'Mean':>12}{'Std':>12}{'Min':>12}{'Max':>12}{'Best':>12}",
        "-" * 75,
    ]
    for i, lbl in enumerate(labels):
        lines.append(
            f"{lbl:<10}{mean_c[i]:>12.6f}{std_c[i]:>12.6f}"
            f"{min_c[i]:>12.6f}{max_c[i]:>12.6f}{best_c[i]:>12.6f}"
        )

    lines += [
        "",
        "CONVERGENCE INTERPRETATION",
        "-" * 75,
        "Low σ/|mean| ratio → coefficient consistently found → likely meaningful",
        "High σ/|mean| ratio → coefficient varies a lot → optimizer indifferent",
        "",
        f"{'Mode':<10}{'|CV| = σ/|mean|':>20}{'Interpretation':>30}",
        "-" * 75,
    ]
    for i, lbl in enumerate(labels):
        if abs(mean_c[i]) > 1e-6:
            cv = abs(std_c[i] / mean_c[i])
            interp = "ROBUST" if cv < 0.1 else ("MODERATE" if cv < 0.4 else "VARIABLE")
        else:
            cv = float('inf')
            interp = "NEAR ZERO"
        lines.append(f"{lbl:<10}{cv:>20.4f}{interp:>30}")

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"Saved -> {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Visualise batch optimisation boundaries and coefficient statistics.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--results-dir', default='batch_results',
                        help='Directory containing batch_summary.json')
    parser.add_argument('--output-dir',  default=None,
                        help='Where to save figures (default: results-dir)')
    args = parser.parse_args()

    output_dir = args.output_dir or args.results_dir
    os.makedirs(output_dir, exist_ok=True)

    summary, runs = load_runs(args.results_dir)
    n = len(runs)
    print(f"Loaded {n} valid runs from {args.results_dir}")

    geo = summary.get('geometry', {})
    a = geo.get('inscribed_square_half_side', 0.5)
    L = geo.get('bounding_box_half_width',    5.0)

    objs = np.array([r['objective'] for r in runs])
    print(f"Objective  min={objs.min():.6f}  mean={objs.mean():.6f}  "
          f"std={objs.std():.6f}  max={objs.max():.6f}")

    plot_all_boundaries(runs, a, L,
                        os.path.join(output_dir, 'all_boundaries.png'))
    plot_boundary_grid(runs, a,
                       os.path.join(output_dir, 'boundary_grid.png'))
    plot_coefficient_stats(runs,
                           os.path.join(output_dir, 'coefficient_stats.png'))
    write_coefficient_table(runs,
                            os.path.join(output_dir, 'coefficient_table.txt'))

    print("\nDone. Files written:")
    for name in ['all_boundaries.png', 'boundary_grid.png',
                 'coefficient_stats.png', 'coefficient_table.txt']:
        print(f"  {os.path.join(output_dir, name)}")


if __name__ == '__main__':
    main()
