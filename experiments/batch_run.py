"""
Batch runner: execute the shape optimization N times with different random seeds.

Geometry:
  - Bounding box:      [-L, L] x [-L, L]  (L = 5.0, set in config)
  - Inscribed square:  [-a, a] x [-a, a]  (a = 0.5, set in config)
  - Initial guess:     circle with radius R = 2.0  (set in config)

Results are saved to:
  <output_dir>/
    run_00_seed0/     <- per-run output (summary.txt, progress plot, etc.)
    run_01_seed1/
    ...
    batch_summary.json    <- machine-readable summary of all runs

Usage:
  python experiments/batch_run.py
  python experiments/batch_run.py --n-runs 30 --maxiter 50 --workers -1
  python experiments/batch_run.py --resume    # skip already-completed runs
"""

import sys
import os
import json
import argparse
import traceback
import logging
from datetime import datetime

import numpy as np

# ---------------------------------------------------------------------------
# Path setup — works whether called from repo root or from experiments/
# ---------------------------------------------------------------------------
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(_here)
_src  = os.path.join(_root, 'src')
for p in [_root, _src]:
    if p not in sys.path:
        sys.path.insert(0, p)

from optimization.config import OPTIMIZATION_CONFIG, SIMULATION_CONFIG, INITIAL_GUESS
from optimization.run_optimization import run_optimization


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _setup_logging(log_path: str) -> logging.Logger:
    logger = logging.getLogger('batch_run')
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s  %(levelname)-7s  %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S')
    fh = logging.FileHandler(log_path, mode='a')
    fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.handlers = []
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def _load_existing_summary(summary_path: str) -> dict:
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            return json.load(f)
    return None


def _save_summary(summary: dict, path: str):
    with open(path, 'w') as f:
        json.dump(summary, f, indent=2)


# ---------------------------------------------------------------------------
# Core batch runner
# ---------------------------------------------------------------------------

def run_batch(
    n_runs:      int  = 30,
    maxiter:     int  = 50,
    workers:     int  = -1,
    output_dir:  str  = 'batch_results',
    seeds:       list = None,
    resume:      bool = False,
) -> dict:
    """
    Run the shape optimization `n_runs` times with distinct random seeds.

    Parameters
    ----------
    n_runs      : Number of independent optimization runs.
    maxiter     : Max DE iterations per run.
    workers     : DE parallel workers (-1 = all CPU cores).
    output_dir  : Root directory for all run outputs.
    seeds       : Explicit list of seeds (default: 0 .. n_runs-1).
    resume      : If True, skip runs whose output directory already exists.
    """
    if seeds is None:
        seeds = list(range(n_runs))
    assert len(seeds) == n_runs, "len(seeds) must equal n_runs"

    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, 'batch_summary.json')
    log_path     = os.path.join(output_dir, 'batch_run.log')
    log          = _setup_logging(log_path)

    # ------------------------------------------------------------------
    # Load or initialise the summary dict
    # ------------------------------------------------------------------
    existing = _load_existing_summary(summary_path) if resume else None
    if existing:
        batch_summary = existing
        completed_ids = {r['run_id'] for r in batch_summary['runs']}
        log.info(f"RESUME mode: {len(completed_ids)} runs already completed.")
    else:
        batch_summary = {
            'n_runs':     n_runs,
            'maxiter':    maxiter,
            'workers':    workers,
            'start_time': datetime.now().isoformat(),
            'geometry': {
                'bounding_box_half_width': OPTIMIZATION_CONFIG['L'],
                'inscribed_square_half_side': OPTIMIZATION_CONFIG['a'],
                'initial_circle_radius': INITIAL_GUESS['R'],
            },
            'runs': [],
        }
        completed_ids = set()

    # ------------------------------------------------------------------
    # Run loop
    # ------------------------------------------------------------------
    for i, seed in enumerate(seeds):
        if i in completed_ids:
            log.info(f"[{i+1:02d}/{n_runs}] seed={seed}  SKIPPED (already done)")
            continue

        log.info(f"[{i+1:02d}/{n_runs}] seed={seed}  starting ...")
        run_dir = os.path.join(output_dir, f'run_{i:02d}_seed{seed}')

        # Build per-run config overrides
        opt_config = OPTIMIZATION_CONFIG.copy()
        opt_config['seed']        = seed
        opt_config['output_dir']  = run_dir
        opt_config['maxiter']     = maxiter
        opt_config['workers']     = workers
        opt_config['save_frames'] = False          # no per-frame PNGs in batch
        opt_config['viz_interval'] = maxiter + 100  # effectively disables in-loop viz

        run_record = {
            'run_id':     i,
            'seed':       seed,
            'output_dir': run_dir,
            'start_time': datetime.now().isoformat(),
        }

        try:
            result, tracker = run_optimization(
                opt_config=opt_config,
                sim_params=SIMULATION_CONFIG.copy(),
                initial_guess=INITIAL_GUESS.copy(),
                verbose=False,
            )

            run_record.update({
                'success':      bool(result.success),
                'objective':    float(result.fun),
                'coefficients': result.x.tolist(),
                'n_iterations': int(result.nit),
                'n_evals':      int(result.nfev),
                'message':      result.message,
                'end_time':     datetime.now().isoformat(),
            })
            log.info(
                f"[{i+1:02d}/{n_runs}] seed={seed}  "
                f"obj={result.fun:.6f}  success={result.success}"
            )

        except Exception as exc:  # noqa: BLE001
            run_record.update({
                'success':   False,
                'error':     str(exc),
                'traceback': traceback.format_exc(),
                'end_time':  datetime.now().isoformat(),
            })
            log.error(f"[{i+1:02d}/{n_runs}] seed={seed}  FAILED: {exc}")

        batch_summary['runs'].append(run_record)

        # Incremental save after every run so nothing is lost on crash
        _save_summary(batch_summary, summary_path)

    # ------------------------------------------------------------------
    # Final aggregate statistics
    # ------------------------------------------------------------------
    successful = [r for r in batch_summary['runs']
                  if r.get('success') and 'objective' in r]

    batch_summary['end_time']    = datetime.now().isoformat()
    batch_summary['n_successful'] = len(successful)

    if successful:
        objs = np.array([r['objective'] for r in successful])
        best_idx = int(np.argmin(objs))
        batch_summary['statistics'] = {
            'min':    float(objs.min()),
            'max':    float(objs.max()),
            'mean':   float(objs.mean()),
            'std':    float(objs.std()),
            'median': float(np.median(objs)),
        }
        batch_summary['best_run'] = {
            'run_id':       successful[best_idx]['run_id'],
            'seed':         successful[best_idx]['seed'],
            'objective':    float(objs[best_idx]),
            'coefficients': successful[best_idx]['coefficients'],
        }

    _save_summary(batch_summary, summary_path)

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------
    log.info("=" * 60)
    log.info(f"BATCH COMPLETE: {len(successful)}/{n_runs} successful")
    if successful:
        s = batch_summary['statistics']
        log.info(f"  Best  obj : {s['min']:.6f}")
        log.info(f"  Mean  obj : {s['mean']:.6f}  ±  {s['std']:.6f}")
        log.info(f"  Worst obj : {s['max']:.6f}")
        log.info(f"  Best run  : run_{batch_summary['best_run']['run_id']:02d}_seed{batch_summary['best_run']['seed']}")
    log.info(f"  Summary   : {summary_path}")
    log.info("=" * 60)

    return batch_summary


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Run shape optimisation N times with different random seeds.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--n-runs',     type=int, default=30,
                        help='Number of independent optimisation runs')
    parser.add_argument('--maxiter',    type=int, default=50,
                        help='Max differential-evolution iterations per run')
    parser.add_argument('--workers',    type=int, default=-1,
                        help='DE parallel workers (-1 = all CPU cores)')
    parser.add_argument('--output-dir', type=str, default='batch_results',
                        help='Root directory for all run outputs')
    parser.add_argument('--resume',     action='store_true',
                        help='Skip runs whose output directory already exists')
    args = parser.parse_args()

    run_batch(
        n_runs=args.n_runs,
        maxiter=args.maxiter,
        workers=args.workers,
        output_dir=args.output_dir,
        resume=args.resume,
    )


if __name__ == '__main__':
    main()
