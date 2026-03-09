"""
model.py — Manifold-valued time series models.
Contains: Reg, RidgeReg (parametric), and ensemble strategies.
"""

from __future__ import annotations

import abc
from typing import Optional, Union, List
import numpy as np
import jax
import jax.numpy as jnp

from TimeSeries.verification_metrics import errfun
from morphomatics.manifold import Euclidean
from TimeSeries.reg import RidgeRegression, PolyRegression
from morphomatics.opt import RiemannianSteepestDescent
from morphomatics import manifold

__all__ = [
    'EnsembleStrategy',
    'VelocityEnsemble',
    'AdaptiveVelocityEnsemble',
    'MAEnsemble',
    'Reg',
    'RidgeReg',
]

maxiter, mingradnorm = 80, 1e-5


# ============================================================================
# ENSEMBLE STRATEGIES
# ============================================================================

class EnsembleStrategy(abc.ABC):
    """Base class for ensemble/post-processing strategies."""

    @abc.abstractmethod
    def adjust(self, M, y_history, y_pred, y_history_preds=None):
        pass


class VelocityEnsemble(EnsembleStrategy):
    """Blend model prediction with a one-step velocity extrapolation."""

    def __init__(self, alpha: float = 0.5, min_history: int = 2):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.alpha       = alpha
        self.min_history = min_history

    def adjust(self, M, y_history, y_pred, y_history_preds=None):
        if len(y_history) < self.min_history or self.alpha == 0.0:
            return y_pred
        v     = M.metric.log(y_history[-2], y_history[-1])
        y_vel = M.metric.exp(y_history[-1], v)
        return M.metric.geopoint(y_pred, y_vel, self.alpha)


class AdaptiveVelocityEnsemble(EnsembleStrategy):
    """Velocity ensemble with alpha adapted from recent prediction quality."""

    def __init__(self, alpha_range: tuple = (0.2, 0.8), window: int = 3):
        self.alpha_min, self.alpha_max = alpha_range
        self.window        = window
        self.recent_errors = []

    def adjust(self, M, y_history, y_pred, y_history_preds=None):
        if len(y_history) < 2:
            return y_pred
        if y_history_preds is not None and len(y_history_preds) > 0:
            self.update_error(M.metric.dist(y_history[-1], y_history_preds[-1]))
        alpha    = self._compute_alpha()
        velocity = M.metric.log(y_history[-2], y_history[-1])
        vel_pred = M.metric.exp(y_history[-1], velocity)
        return M.metric.geopoint(y_pred, vel_pred, alpha)

    def _compute_alpha(self):
        if len(self.recent_errors) < 2:
            return (self.alpha_min + self.alpha_max) / 2
        recent = self.recent_errors[-self.window:]
        trend  = np.mean(np.diff(recent))
        return self.alpha_min if trend < 0 else self.alpha_max

    def update_error(self, error: float):
        self.recent_errors.append(error)
        if len(self.recent_errors) > self.window * 2:
            self.recent_errors.pop(0)


class MAEnsemble(EnsembleStrategy):
    """Moving-average ensemble: correct prediction using weighted past residuals."""

    def __init__(self, alpha: float = 0.5, order: int = 2,
                 weights: np.ndarray = None, min_history: int = 3):
        self.alpha       = alpha
        self.order       = order
        self.min_history = min_history
        if weights is not None:
            self.weights = np.array(weights)
        else:
            w            = np.array([np.exp(i * 0.5) for i in range(order)])
            self.weights = w / w.sum()

    def adjust(self, M, y_history, y_pred, y_history_preds=None):
        if y_history_preds is None or len(y_history_preds) < self.min_history:
            return y_pred
        q           = min(self.order, len(y_history_preds))
        past_actuals = y_history[-q:]
        past_preds   = y_history_preds[-q:]
        residuals    = [M.metric.log(p, a) for p, a in zip(past_preds, past_actuals)]
        v_fix        = _linear_combination(M, y_pred, past_preds, residuals, self.weights[-q:])
        return M.metric.exp(y_pred, self.alpha * v_fix)


# ============================================================================
# BASE MODEL
# ============================================================================

class Model(abc.ABC):
    """Base class for manifold-valued time series models."""

    def __init__(self, M, lag=None, x_fit=None, y_fit=None, ensemble_strategy=None):
        self.M                 = M
        self.lag               = lag
        self.x_fit             = x_fit
        self.y_fit             = y_fit
        self.ensemble_strategy = ensemble_strategy
        self.weight            = None
        self.param             = None

    @property
    def parametric(self) -> bool:
        return True

    def set_ensemble_strategy(self, strategy):
        self.ensemble_strategy = strategy
        return self

    def predict(self, X: np.ndarray, iterative: bool = False) -> np.ndarray:
        if not iterative:
            return self.model_fun(X)

        if self.y_fit is None:
            raise ValueError("Model must be fitted before prediction.")

        original_y_fit = self.y_fit
        original_x_fit = self.x_fit
        working_y      = self.y_fit.copy()
        working_x      = self.x_fit.copy() if self.x_fit is not None else None
        predictions    = []

        for i in range(len(X)):
            self.y_fit = working_y
            self.x_fit = working_x
            next_pred  = self.model_fun(X[i:i + 1])[0]
            predictions.append(next_pred)
            working_y = np.concatenate([working_y, [next_pred]], axis=0)
            if working_x is not None:
                working_x = np.concatenate([working_x, X[i:i + 1]], axis=0)

        self.y_fit = original_y_fit
        self.x_fit = original_x_fit
        return np.array(predictions)

    def validate(self, X: np.ndarray, y: np.ndarray):
        if len(X) != len(y):
            raise ValueError(f"X and y must have same length.")
        lag, n = self.lag, len(y)
        m      = max(0, n - lag) if lag and lag > 0 else 0
        self.x_fit, self.y_fit = X[m:], y[m:]

    @abc.abstractmethod
    def fit(self, X, y):
        pass

    @abc.abstractmethod
    def model_fun(self, X):
        pass


# ============================================================================
# PARAMETRIC MODELS
# ============================================================================

class Reg(Model):
    """Polynomial (Bézier) regression on manifolds — OLS."""

    def __init__(self, M: manifold.Manifold, lag: Union[bool, int] = False, degree: int = 3):
        self.trend  = None
        self.degree = degree
        lag         = degree + 1 if lag else lag
        super().__init__(M=M, lag=lag)

    def fit(self, X: np.ndarray, y: np.ndarray, P_init=None) -> 'Reg':
        self.validate(X, y)
        self.trend = PolyRegression(
            self.M, self.y_fit, self.x_fit, self.degree,
            P_init=P_init, maxiter=30, mingradnorm=1e-6
        ).trend
        return self

    @property
    def control_points(self):
        """Fitted control points, usable as warm start for RidgeReg."""
        return self.trend.control_points if self.trend is not None else None

    def model_fun(self, X: np.ndarray) -> np.ndarray:
        if self.trend is None:
            raise ValueError("Model must be fitted before prediction.")
        return jax.vmap(self.trend.eval)(X)


class RidgeReg(Model):
    """Polynomial ridge regression on manifolds with Mahalanobis regularization."""

    def __init__(self, M: manifold.Manifold, mean: np.ndarray, cov: np.ndarray,
                 ridge_const: float, lag: Union[bool, int] = False, degree: int = 3):
        self.trend       = None
        self.mean        = mean
        self.cov         = cov
        self.ridge_const = ridge_const
        self.degree      = degree
        lag              = degree + 1 if lag else lag
        super().__init__(M, lag)

    def fit(self, X: np.ndarray, y: np.ndarray, P_init=None) -> 'RidgeReg':
        self.validate(X, y)
        self.trend = RidgeRegression(
            M           = self.M,
            Y           = self.y_fit,
            param       = self.x_fit,
            mean        = self.mean,
            cov         = self.cov,
            ridge_const = self.ridge_const,
            degree      = self.degree,
            P_init      = P_init,
            maxiter     = 30,
            mingradnorm = 1e-6
        ).trend
        return self

    @property
    def control_points(self):
        """Fitted control points, usable as warm start for next fit."""
        return self.trend.control_points if self.trend is not None else None

    def model_fun(self, X: np.ndarray) -> np.ndarray:
        if self.trend is None:
            raise ValueError("Model must be fitted before prediction.")
        return jax.vmap(self.trend.eval)(X)


# ============================================================================
# UTILITY
# ============================================================================

def _linear_combination(M, ref, points, vectors, weights) -> jnp.ndarray:
    result = jnp.zeros(M.point_shape)
    for pt, vec, w in zip(points, vectors, weights):
        result = result + w * M.metric.transp(pt, ref, vec)
    return result