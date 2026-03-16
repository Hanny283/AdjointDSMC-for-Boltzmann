"""
Analyse and compare results from a batch optimisation run.

Produces:
  <output_dir>/batch_analysis.png   — 6-panel summary figure
  <output_dir>/batch_report.txt     — human-readable statistics report

Usage:
  python experiments/compare_results.py
  python experiments/compare_results.py --results-dir batch_results
  python experiments/compare_results.py --results-dir batch_results --output-dir figs/
"""

import sys
import os
import json
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')          # headless-safe backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import LineCollection

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
_src  = os.path.join(_root, 'src')
for p in [_root, _src]:
    if p not in sys.path:
        sys.path.insert(0, p)


# ---------------------------------------------------------------------------
# Fourier shape helper (mirrors shape_optimizer.compute_radius)
# ---------------------------------------------------------------------------

def _fourier_radius(theta: np.ndarray, c: np.ndarray) -> np.ndarray:
    M = (len(c) - 1) // 2
    r = np.full_like(theta, c[0])
    for m in range(1, M + 1):
        r = r + c[m] * np.cos(m * theta) + c[M + m] * np.sin(m * theta)
    return r


def _shape_xy(c: np.ndarray, n: int = 400):
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    r = _fourier_radius(theta, c)
    return r * np.cos(theta), r * np.sin(theta)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_summary(results_dir: str) -> dict:
    path = os.path.join(results_dir, 'batch_summary.json')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No batch_summary.json in '{results_dir}'. "
            "Run experiments/batch_run.py first."
        )
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Text report
# ---------------------------------------------------------------------------

def _write_report(path: str, summary: dict, successful: list,
                  objectives: np.ndarray, coefficients: np.ndarray,
                  converged_mask: np.ndarray, threshold: float):
    lines = [
        "BATCH OPTIMISATION REPORT",
        "=" * 60,
        f"Generated : {__import__('datetime').datetime.now()}",
        f"Results   : {summary.get('start_time', 'N/A')} -> {summary.get('end_time', 'N/A')}",
        "",
        "GEOMETRY",
        "-" * 40,
    ]
    geo = summary.get('geometry', {})
    lines += [
        f"  Bounding box half-width (L)  : {geo.get('bounding_box_half_width', 'N/A')}",
        f"  Inscribed square half-side(a): {geo.get('inscribed_square_half_side', 'N/A')}",
        f"  Initial circle radius (R)    : {geo.get('initial_circle_radius', 'N/A')}",
        "",
        "RUN STATISTICS",
        "-" * 40,
        f"  Total runs           : {summary['n_runs']}",
        f"  Successful           : {len(successful)}",
        f"  Failed               : {summary['n_runs'] - len(successful)}",
        "",
        "OBJECTIVE STATISTICS",
        "-" * 40,
        f"  Min    : {objectives.min():.8f}",
        f"  Max    : {objectives.max():.8f}",
        f"  Mean   : {objectives.mean():.8f}",
        f"  Std    : {objectives.std():.8f}",
        f"  Median : {np.median(objectives):.8f}",
        "",
        f"CONVERGENCE  (within {(threshold-1)*100:.1f}% of global min)",
        "-" * 40,
        f"  Converged runs : {converged_mask.sum()} / {len(objectives)}",
        f"  Threshold      : {objectives.min() * threshold:.8f}",
        "",
        "ALL RUNS  (sorted by objective)",
        "-" * 40,
    ]
    for rank, idx in enumerate(np.argsort(objectives)):
        r = successful[idx]
        flag = " <-- BEST" if rank == 0 else ""
        lines.append(
            f"  Rank {rank+1:2d} | obj={objectives[idx]:.8f} | "
            f"seed={r['seed']:3d} | run={r['run_id']:02d} | "
            f"nit={r.get('n_iterations','?')}{flag}"
        )
    best_c = coefficients[np.argmin(objectives)]
    lines += [
        "",
        "BEST RUN COEFFICIENTS",
        "-" * 40,
    ]
    M = (len(best_c) - 1) // 2
    lines.append(f"  c[0]  (base radius) = {best_c[0]:.8f}")
    for m in range(1, M + 1):
        lines.append(f"  c[{m}]  (cos {m}θ)      = {best_c[m]:.8f}")
    for m in range(1, M + 1):
        lines.append(f"  c[{M+m}]  (sin {m}θ)      = {best_c[M+m]:.8f}")

    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"Report saved  -> {path}")


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def _build_figure(
    summary:      dict,
    successful:   list,
    objectives:   np.ndarray,
    coefficients: np.ndarray,
    converged:    np.ndarray,
    threshold:    float,
    a:            float,
) -> plt.Figure:

    n = len(successful)
    M = (coefficients.shape[1] - 1) // 2
    seeds = [r['seed'] for r in successful]
    best_idx = int(np.argmin(objectives))

    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor('#0f1117')
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.38,
                           left=0.07, right=0.96, top=0.91, bottom=0.08)

    text_kw  = dict(color='#e0e0e0')
    title_kw = dict(color='#f0f0f0', fontsize=11, fontweight='bold', pad=8)
    ax_face  = '#1a1d27'
    spine_c  = '#3a3d4a'

    def _style(ax):
        ax.set_facecolor(ax_face)
        for sp in ax.spines.values():
            sp.set_edgecolor(spine_c)
        ax.tick_params(colors='#b0b0b0', labelsize=8)
        ax.xaxis.label.set_color('#c0c0c0')
        ax.yaxis.label.set_color('#c0c0c0')

    # ── 1. Histogram of objectives ─────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    _style(ax1)
    ax1.hist(objectives, bins=min(15, max(5, n // 2)),
             color='#4a9eff', edgecolor='#0f1117', alpha=0.85, linewidth=0.6)
    ax1.axvline(objectives.min(), color='#ff6b6b', lw=1.8, ls='--',
                label=f'Best = {objectives.min():.4f}')
    ax1.axvline(objectives.mean(), color='#ffd166', lw=1.5, ls=':',
                label=f'Mean = {objectives.mean():.4f}')
    ax1.axvline(objectives.min() * threshold, color='#06d6a0', lw=1.2, ls='-.',
                label=f'{(threshold-1)*100:.0f}% threshold')
    ax1.set_xlabel('Final objective value', **text_kw)
    ax1.set_ylabel('Count', **text_kw)
    ax1.set_title('Objective distribution', **title_kw)
    ax1.legend(fontsize=7, facecolor=ax_face, edgecolor=spine_c,
               labelcolor='#c0c0c0')

    # ── 2. Sorted objectives (convergence ladder) ──────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    _style(ax2)
    sorted_obj = np.sort(objectives)
    colors_sorted = np.where(sorted_obj <= objectives.min() * threshold,
                             '#06d6a0', '#4a9eff')
    ax2.scatter(range(n), sorted_obj, c=colors_sorted, s=40, zorder=5,
                edgecolors='none')
    ax2.plot(sorted_obj, color='#4a9eff', lw=1, alpha=0.5)
    ax2.axhline(objectives.min() * threshold, color='#06d6a0', lw=1.2, ls='--',
                label=f'Convergence threshold')
    ax2.set_xlabel('Run rank (sorted)', **text_kw)
    ax2.set_ylabel('Objective value', **text_kw)
    ax2.set_title('Convergence ladder', **title_kw)
    ax2.legend(fontsize=7, facecolor=ax_face, edgecolor=spine_c,
               labelcolor='#c0c0c0')

    # ── 3. Objective vs seed ───────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    _style(ax3)
    sc = ax3.scatter(seeds, objectives, c=objectives, cmap='plasma',
                     s=55, edgecolors='#0f1117', linewidth=0.4, zorder=5)
    cb = fig.colorbar(sc, ax=ax3, pad=0.02)
    cb.ax.tick_params(labelcolor='#b0b0b0', labelsize=7)
    cb.set_label('Objective', color='#c0c0c0', fontsize=8)
    ax3.set_xlabel('Seed', **text_kw)
    ax3.set_ylabel('Final objective', **text_kw)
    ax3.set_title('Objective vs random seed', **title_kw)

    # ── 4. Coefficient box plots ───────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, :2])
    _style(ax4)
    coef_labels = (['c₀'] +
                   [f'c{m}cos' for m in range(1, M + 1)] +
                   [f'c{m}sin' for m in range(1, M + 1)])
    bp = ax4.boxplot(coefficients, labels=coef_labels, patch_artist=True,
                     medianprops=dict(color='#ffd166', lw=1.5),
                     whiskerprops=dict(color='#b0b0b0'),
                     capprops=dict(color='#b0b0b0'),
                     flierprops=dict(marker='.', markersize=4,
                                     markerfacecolor='#ff6b6b'))
    palette = ['#4a9eff'] + ['#74c7ec'] * M + ['#f38ba8'] * M
    for patch, col in zip(bp['boxes'], palette):
        patch.set_facecolor(col)
        patch.set_alpha(0.7)
    # Overlay best-run coefficients as stars
    ax4.plot(range(1, coefficients.shape[1] + 1),
             coefficients[best_idx], 'r*', markersize=9,
             label='Best run', zorder=6)
    ax4.axhline(0, color='#3a3d4a', lw=1)
    ax4.set_title('Fourier coefficient distributions across runs', **title_kw)
    ax4.set_ylabel('Coefficient value', **text_kw)
    ax4.legend(fontsize=8, facecolor=ax_face, edgecolor=spine_c,
               labelcolor='#c0c0c0')
    ax4.tick_params(axis='x', labelsize=8)

    # ── 5. Top-K shapes overlay ────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 2])
    _style(ax5)
    top_k = min(8, n)
    sorted_idx = np.argsort(objectives)[:top_k]
    cmap = plt.cm.cool
    norm = plt.Normalize(0, top_k - 1)
    for rank, idx in enumerate(sorted_idx):
        x, y = _shape_xy(coefficients[idx])
        lw  = 2.2 if rank == 0 else 0.9
        alp = 1.0 if rank == 0 else 0.55
        ax5.plot(x, y, lw=lw, alpha=alp, color=cmap(norm(rank)),
                 label=f'#{rank+1}: {objectives[idx]:.4f}')
    # Bounding box outline (faint)
    L = summary.get('geometry', {}).get('bounding_box_half_width', 5.0)
    ax5.plot([-L, L, L, -L, -L], [-L, -L, L, L, -L],
             lw=0.6, color='#555', ls='--')
    # Inscribed square
    sq = MplPolygon([[-a, -a], [a, -a], [a, a], [-a, a]],
                    closed=True, fill=False, edgecolor='#ff6b6b',
                    linewidth=1.8, linestyle='--')
    ax5.add_patch(sq)
    ax5.set_aspect('equal')
    lim = max(3.0, np.abs(coefficients[sorted_idx[0]]).max() + 0.5)
    ax5.set_xlim(-lim, lim)
    ax5.set_ylim(-lim, lim)
    ax5.legend(fontsize=6.5, facecolor=ax_face, edgecolor=spine_c,
               labelcolor='#c0c0c0', loc='upper right',
               ncol=2 if top_k > 4 else 1)
    ax5.set_title(f'Top-{top_k} converged shapes', **title_kw)
    ax5.set_xlabel('x', **text_kw)
    ax5.set_ylabel('y', **text_kw)

    # ── Super title ────────────────────────────────────────────────────────
    conv_pct = 100 * converged.sum() / n
    fig.suptitle(
        f'Batch Optimisation Analysis  —  {n} successful runs  '
        f'|  {converged.sum()} converged ({conv_pct:.0f}%)',
        fontsize=14, fontweight='bold', color='#f0f0f0', y=0.97
    )
    return fig


# ---------------------------------------------------------------------------
# Main analysis function
# ---------------------------------------------------------------------------

def analyze_batch(results_dir: str = 'batch_results',
                  output_dir:  str = None,
                  convergence_pct: float = 5.0) -> None:
    """
    Load batch_summary.json and produce analysis figure + text report.

    Parameters
    ----------
    results_dir     : Directory that contains batch_summary.json.
    output_dir      : Where to save outputs (defaults to results_dir).
    convergence_pct : A run is "converged" if its objective is within this
                      percentage of the global best (default 5 %).
    """
    if output_dir is None:
        output_dir = results_dir
    os.makedirs(output_dir, exist_ok=True)

    summary    = load_summary(results_dir)
    runs       = summary['runs']
    # Accept any run that produced a finite objective, regardless of scipy's
    # convergence flag (result.success is False when maxiter is reached even
    # if a good solution was found, which is common with stochastic objectives).
    successful = [r for r in runs
                  if 'objective' in r and r['objective'] < 1e9]
    failed     = [r for r in runs if r not in successful]

    print(f"\n{'='*60}")
    print("BATCH RESULTS ANALYSIS")
    print(f"{'='*60}")
    print(f"Total runs  : {len(runs)}")
    print(f"Successful  : {len(successful)}")
    print(f"Failed      : {len(failed)}")

    if not successful:
        print("No successful runs found. Nothing to plot.")
        return

    objectives   = np.array([r['objective']    for r in successful])
    coefficients = np.array([r['coefficients'] for r in successful])
    threshold    = 1.0 + convergence_pct / 100.0
    converged    = objectives <= objectives.min() * threshold

    a = summary.get('geometry', {}).get('inscribed_square_half_side', 0.5)

    print(f"\nObjective  min={objectives.min():.6f}  "
          f"mean={objectives.mean():.6f}  "
          f"std={objectives.std():.6f}  "
          f"max={objectives.max():.6f}")
    print(f"Converged  : {converged.sum()} / {len(objectives)} "
          f"(within {convergence_pct:.1f}% of best)")

    fig = _build_figure(summary, successful, objectives, coefficients,
                        converged, threshold, a)

    fig_path    = os.path.join(output_dir, 'batch_analysis.png')
    report_path = os.path.join(output_dir, 'batch_report.txt')

    fig.savefig(fig_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\nFigure saved -> {fig_path}")

    _write_report(report_path, summary, successful, objectives, coefficients,
                  converged, threshold)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Analyse results from a batch shape-optimisation run.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--results-dir',    type=str,   default='batch_results',
                        help='Directory containing batch_summary.json')
    parser.add_argument('--output-dir',     type=str,   default=None,
                        help='Where to save figures/reports (default: results-dir)')
    parser.add_argument('--convergence-pct', type=float, default=5.0,
                        help='% above global min that counts as "converged"')
    args = parser.parse_args()

    analyze_batch(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        convergence_pct=args.convergence_pct,
    )


if __name__ == '__main__':
    main()
