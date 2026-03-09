"""
Reproducible synthetic experiment for Table 1.

Design:
  - 3 template curves (Geo, Poly, Else), 10 noisy trajectories each = 30 total
  - Pooled and randomly shuffled before splitting
  - Split: 20 train / 10 validation / 5 test
  - Degree-3 Bézier polynomials (n=4 control points)
  - Covariance: sample covariance + 1e-6 * I (diagonal loading)
  - Grid search over lambda on validation set (OLS warm-start)
  - Report mean ± std MAE over 5 test trajectories

Run from the project root (where TimeSeries/ and util_pred.py are located).
"""

import numpy as np
import numpy.linalg as lg
import jax
from morphomatics.manifold import Sphere, PowerManifold
from morphomatics.stats import ExponentialBarycenter
from TimeSeries.verification_metrics import errfun
from TimeSeries.stats import sph_correlated_trjs
from TimeSeries.main import pred, pred_grid_search
from TimeSeries.model import Reg, RidgeReg
from util_pred import cov_mat, fit_poly_dc

# ── reproducibility ───────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

# ── manifold setup ────────────────────────────────────────────────────────────
M   = Sphere()
Ex  = ExponentialBarycenter()
err = errfun(M.metric.dist)

# ── experiment parameters ─────────────────────────────────────────────────────
DEGREE      = 3           # polynomial degree (n_cp = 4 control points)
N_PER_TEMPL = 10          # noisy trajectories per template
TEMPLATES   = ['Geo', 'Poly', 'Else']
N_TOTAL     = N_PER_TEMPL * len(TEMPLATES)  # 30
N_TRAIN     = 20
N_VAL       = 5          # larger validation set for stable lambda selection
N_TEST      = 5
assert N_TRAIN + N_VAL + N_TEST == N_TOTAL

N_POINTS    = 45
NOISE_STD   = 0.05
LON_MAX     = 0.75 * np.pi
LAT_MAX     = np.pi / 20
DIAG_LOAD   = 1e-6        # diagonal loading for covariance regularization

LAMBDA_GRID = [0, 1e-6, 1e-4, 1e-2, 5e-2, 0.1, 0.2, 0.4]

PRED_ARGS = {
    'n_learn':   DEGREE + 1,  # = 4, consistent with notebook
    'n_pred':    1,
    'iterative': False,
}


def main():
    print(f"Synthetic experiment: {len(TEMPLATES)} templates x {N_PER_TEMPL} trajectories each")
    print(f"Split: {N_TRAIN} train / {N_VAL} val / {N_TEST} test  |  degree={DEGREE}  |  seed={SEED}")

    # ── generate data from all 3 templates ───────────────────────────────────
    Y_all = []
    for template in TEMPLATES:
        Y_t, _ = sph_correlated_trjs(
            LON_MAX, LAT_MAX,
            n_trj=N_PER_TEMPL, n_points=N_POINTS,
            noise_std=NOISE_STD, mean_curve=template
        )
        Y_all.extend(Y_t)

    # ── shuffle and split ─────────────────────────────────────────────────────
    idx    = np.random.permutation(N_TOTAL)
    Y_all  = [Y_all[i] for i in idx]

    Y_train = Y_all[:N_TRAIN]
    Y_val   = Y_all[N_TRAIN:N_TRAIN + N_VAL]
    Y_test  = Y_all[N_TRAIN + N_VAL:]
    assert len(Y_test) == N_TEST

    # ── fit Bézier polynomials to training trajectories ───────────────────────
    n_cp = DEGREE + 1  # = 4
    dim  = 3           # ambient dim of S^2
    P    = PowerManifold(M, n_cp)
    B_train, _ = fit_poly_dc(M, Y_train, deg=DEGREE)

    # ── covariance: sample cov + diagonal loading ─────────────────────────────
    mean_b = Ex.compute(P, B_train, max_iter=30)
    cov_b  = cov_mat(P.metric.log, B_train, mean_b) + DIAG_LOAD * np.eye(n_cp * dim)

    eigenvalues = np.sort(lg.eigvalsh(cov_b))[::-1]
    print(f"\nCovariance eigenvalues: {np.round(eigenvalues, 6)}")
    print(f"Condition number: {eigenvalues[0] / eigenvalues[-1]:.1f}")

    # ── OLS baseline on test set ──────────────────────────────────────────────
    model_ols = Reg(M, lag=True, degree=DEGREE)
    _, metrics_ols = pred(Y_test, model_ols, **PRED_ARGS, prnt=False)

    mae_ols_per_track = np.array([np.mean(m) for m in metrics_ols['MAE']])
    mae_ols_mean = np.mean(mae_ols_per_track)
    mae_ols_std  = np.std(mae_ols_per_track)
    print(f"\nOLS  MAE: {mae_ols_mean:.4f} +/- {mae_ols_std:.4f}")

    # ── grid search on validation set (OLS warm-start) ────────────────────────
    print(f"\nGrid search over lambda (warm-started from OLS):")
    ols_for_search = Reg(M, lag=True, degree=DEGREE)
    ridge_model_fn = lambda lam: RidgeReg(M, mean_b, cov_b, lam, lag=True, degree=DEGREE)
    val_results    = pred_grid_search(
        Y_val, ols_for_search, ridge_model_fn, LAMBDA_GRID, **PRED_ARGS, prnt=True
    )
    val_maes = {lam: val_results[lam]['mae'] for lam in LAMBDA_GRID}
    lam_star = min(val_maes, key=val_maes.get)
    print(f"Optimal lambda*: {lam_star}")

    # ── ridge regression with lambda* on test set ─────────────────────────────
    if lam_star == 0:
        model_ridge = Reg(M, lag=True, degree=DEGREE)
    else:
        model_ridge = RidgeReg(M, mean_b, cov_b, lam_star, lag=True, degree=DEGREE)

    _, metrics_ridge = pred(Y_test, model_ridge, **PRED_ARGS, prnt=False)

    mae_ridge_per_track = np.array([np.mean(m) for m in metrics_ridge['MAE']])
    mae_ridge_mean = np.mean(mae_ridge_per_track)
    mae_ridge_std  = np.std(mae_ridge_per_track)
    improvement    = 100 * (mae_ols_mean - mae_ridge_mean) / mae_ols_mean

    print(f"Ridge MAE: {mae_ridge_mean:.4f} +/- {mae_ridge_std:.4f}")
    print(f"Improvement: {improvement:.1f}%")

    # ── summary table ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Table 1 -- MAE (radians), mixed-template synthetic experiment")
    print(f"{'='*60}")
    print(f"{'Method':<20} {'MAE (mean +/- std)':<25} {'lambda*'}")
    print(f"{'-'*60}")
    print(f"{'OLS (lambda=0)':<20} {mae_ols_mean:.4f} +/- {mae_ols_std:.4f}")
    print(f"{'Ridge (proposed)':<20} {mae_ridge_mean:.4f} +/- {mae_ridge_std:.4f}         {lam_star}")
    print(f"{'-'*60}")
    print(f"Improvement: {improvement:.1f}%")
    print(f"{'='*60}")
    print(f"\nRandom seed: {SEED}")
    print(f"Templates: {TEMPLATES}, {N_PER_TEMPL} trajectories each")
    print(f"Split: {N_TRAIN} train / {N_VAL} val / {N_TEST} test")
    print(f"Covariance: sample + {DIAG_LOAD} * I (diagonal loading)")


if __name__ == '__main__':
    main()