import numpy as np
import jax
from typing import Optional, List, Tuple, Dict


def pred(
        Y_test: List[np.ndarray],
        model,
        n_learn: int = 1,
        n_pred: int = 1,
        iterative: bool = False,
        ensemble_strategy=None,
        prnt: bool = True
) -> Tuple[List[np.ndarray], dict]:
    """
    Predict manifold-valued trajectories using a fitted model.

    Parameters
    ----------
    Y_test : list of arrays, each shape (n_obs, point_dim)
    model  : fitted model with fit() and predict() methods
    n_learn: minimum observations before first prediction
    n_pred : steps ahead to predict
    iterative : whether to use iterative (closed-loop) prediction
    ensemble_strategy : optional post-processing strategy
    prnt   : print summary if True

    Returns
    -------
    Y_pred  : list of predicted arrays
    metrics : dict with keys 'Y_pred', 'MAE', 'mae'
                - MAE : list of per-track error arrays
                - mae : float, mean over all forecasts
    """
    n_test   = len(Y_test)
    M        = model.M
    dist_fn  = jax.jit(jax.vmap(M.metric.dist, in_axes=(0, 0)))
    strategy = ensemble_strategy if ensemble_strategy is not None else model.ensemble_strategy

    Y_pred = [None] * n_test
    MAE    = [None] * n_test

    for k in range(n_test):
        y_test      = Y_test[k]
        n_forecasts = len(y_test) - n_learn - n_pred + 1
        y_pred      = np.empty((n_forecasts,) + M.point_shape)
        y_preds_track = []

        for n in range(n_learn, len(y_test) - n_pred + 1):
            y_learn = y_test[:n]
            len_x   = n + n_pred
            x       = np.linspace(0.0, 1.0, len_x)

            model        = model.fit(x[:n], y_learn)
            p            = np.array(model.predict(x[n:len_x], iterative=iterative))
            target_pred  = p[n_pred - 1]

            if strategy is not None:
                history_length = getattr(strategy, 'min_history', 2)
                if len(y_learn) >= history_length:
                    target_pred = strategy.adjust(
                        model.M,
                        y_learn[-history_length:],
                        target_pred,
                        np.array(y_preds_track)
                    )

            y_pred[n - n_learn] = target_pred
            y_preds_track.append(target_pred)

        y_true     = y_test[n_learn + n_pred - 1:]
        MAE[k]     = dist_fn(y_true, y_pred)
        Y_pred[k]  = y_pred

    all_mae = np.concatenate(MAE)
    mae     = float(np.mean(all_mae))

    if prnt:
        print('=' * 60)
        print(f'MAE: {mae:.4f}')
        print('=' * 60)

    metrics = {
        'MAE': MAE,
        'mae': mae,
    }
    return Y_pred, metrics


def pred_NHC(
        Y_test: List[np.ndarray],
        model,
        n_learn: int = 1,
        n_pred: int = 2,
        iterative: bool = False,
        ensemble_strategy=None,
        W_test: Optional[List[np.ndarray]] = None,
        prnt: bool = True
) -> Tuple[List[np.ndarray], dict]:
    """
    Predict hurricane trajectories with optional NHC intensity filtering.

    Parameters
    ----------
    Y_test  : list of track arrays, each shape (n_obs, point_dim)
    model   : fitted model with fit() and predict() methods
    n_learn : minimum observations before first prediction
    n_pred  : steps ahead to predict (2 = 12 h for 6 h data)
    W_test  : optional list of wind-speed arrays for NHC filtering
    prnt    : print summary if True

    Returns
    -------
    Y_pred  : list of predicted arrays
    metrics : dict with keys:
                - MAE          : list of per-track error arrays
                - mae          : float, mean over all forecasts
                - nhc_mae      : float or None, NHC-filtered MAE
    """
    n_test      = len(Y_test)
    M           = model.M
    dist_fn     = jax.jit(jax.vmap(M.metric.dist, in_axes=(0, 0)))
    strategy    = ensemble_strategy if ensemble_strategy is not None else model.ensemble_strategy
    TC_THRESHOLD = 34.5

    Y_pred  = [None] * n_test
    MAE     = [None] * n_test
    NHC_MAE = [None] * n_test if W_test is not None else None

    for k in range(n_test):
        y_test      = Y_test[k]
        n_forecasts = len(y_test) - n_learn - n_pred + 1
        y_pred      = np.empty((n_forecasts,) + M.point_shape)

        for n in range(n_learn, len(y_test) - n_pred + 1):
            y_learn = y_test[:n]
            len_x   = n + n_pred
            x       = np.linspace(0.0, 1.0, len_x)

            model       = model.fit(x[:n], y_learn)
            p           = np.array(model.predict(x[n:len_x], iterative=iterative))
            target_pred = p[n_pred - 1]

            if strategy is not None:
                history_length = getattr(strategy, 'min_history', 2)
                if len(y_learn) >= history_length:
                    target_pred = strategy.adjust(
                        model.M,
                        y_learn[-history_length:],
                        target_pred
                    )

            y_pred[n - n_learn] = target_pred

        y_true    = y_test[n_learn + n_pred - 1:]
        mae       = dist_fn(y_true, y_pred)
        MAE[k]    = mae
        Y_pred[k] = y_pred

        # NHC filtering
        if W_test is not None:
            w_test          = W_test[k]
            initial_indices = np.arange(n_learn - 1, min(len(y_test) - n_pred, len(w_test)))
            verify_indices  = initial_indices + n_pred
            valid_mask      = verify_indices < len(w_test)
            initial_indices = initial_indices[valid_mask]
            verify_indices  = verify_indices[valid_mask]

            if len(initial_indices) > 0:
                nhc_mask       = ((w_test[initial_indices] >= TC_THRESHOLD) &
                                  (w_test[verify_indices]  >= TC_THRESHOLD))
                NHC_MAE[k]     = mae[nhc_mask[:len(mae)]]
            else:
                NHC_MAE[k] = np.array([])

    all_mae  = np.concatenate(MAE)
    mae_mean = float(np.mean(all_mae))

    nhc_mae_mean = None
    if W_test is not None:
        all_nhc = np.concatenate([x for x in NHC_MAE if len(x) > 0])
        if len(all_nhc) > 0:
            nhc_mae_mean = float(np.mean(all_nhc))

    if prnt:
        print('=' * 60)
        print(f'MAE:     {mae_mean:.4f}')
        if nhc_mae_mean is not None:
            print(f'NHC MAE: {nhc_mae_mean:.4f}')
        print('=' * 60)

    metrics = {
        'MAE':     MAE,
        'mae':     mae_mean,
        'nhc_mae': nhc_mae_mean,
    }
    return Y_pred, metrics


def pred_grid_search(
        Y_val, model_fn, lambda_grid,
        n_learn=1, n_pred=1, iterative=False, prnt=True
):
    results = {}
    for lam in lambda_grid:
        model = model_fn(lam)
        _, metrics = pred(Y_val, model, n_learn=n_learn, n_pred=n_pred,
                          iterative=iterative, prnt=False)
        results[lam] = metrics
        if prnt:
            print(f'  lambda={lam:.0e}  val MAE={metrics["mae"]:.4f}')
    return results