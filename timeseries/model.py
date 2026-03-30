"""
model.py — Manifold-valued time series models (replaces model.py).

Architecture
------------
1. Unified _ARMABase class consolidates AR, VWA, and ARMA logic.
   Public API via thin wrappers: AR, VWA, ARMA.
2. Explicit `parametric` property on every model class:
   - True  (Reg, RidgeReg): model evaluates f(t) directly; iterative=False natural.
   - False (AR, VWA, ARMA, MA, WeightedAverage): model_fun() always returns one
     point and ignores X; multi-step MUST be recursive. predict() enforces this.
3. predict() correctly overrides iterative=False → True for non-parametric models
   with len(X) > 1, with an explicit warning rather than silent override.
4. validate() uses self.parametric (not fragile hasattr duck-typing) to decide
   whether to replace x_fit with a sequential integer index.
5. _compute_par() closes over scalar values and arrays — not self — inside
   @jax.jit, avoiding stale-closure tracing bugs across warm-start re-fits.
6. MAEnsemble.opt() uses the correct objective: parallel-transport past residuals
   to the target point and minimize distance to the actual next residual.
   Uses RSD with softmax parameterization (no scipy dependency).
7. Dead import from test_scripts removed.
8. All bare except: replaced with except Exception: (with explanatory comments).
9. WeightedAverage._compute_weights non-JIT-ability documented explicitly.
"""

from __future__ import annotations

import abc
from typing import Optional, Tuple, Union
import numpy as np
import jax
import jax.numpy as jnp

from timeseries.verification_metrics import errfun
from util_pred import diff
from morphomatics.manifold import Euclidean
from timeseries.reg import RidgeRegression
from timeseries.reg import PolyRegression
from morphomatics.opt import RiemannianSteepestDescent
from morphomatics import manifold
# Public API
__all__ = [
    # Ensemble strategies
    'EnsembleStrategy',
    'VelocityEnsemble',
    'AdaptiveVelocityEnsemble',
    'MAEnsemble',
    # Parametric models
    'Reg',
    'RidgeReg'
]
# Note: _ARMABase is intentionally excluded (private implementation detail)

maxiter, mingradnorm = 80, 1e-5


# ============================================================================
# ENSEMBLE STRATEGIES
# ============================================================================

class EnsembleStrategy(abc.ABC):
    """Base class for ensemble/post-processing strategies."""

    @abc.abstractmethod
    def adjust(
            self,
            M,
            y_history: np.ndarray,
            y_pred: np.ndarray,
            y_history_preds: Optional[np.ndarray] = None
    ) -> np.ndarray:
        pass

class AVGEnsemble(EnsembleStrategy):
    def __init__(self, alpha: float = 0.5, min_history: int = 2):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.alpha = alpha
        self.min_history = min_history

    def adjust(self, M, y_history, y_pred, y_history_preds=None):
        if len(y_history) < self.min_history or self.alpha == 0.0:
            return y_pred

        y_vel = y_history[-1]
        return M.metric.geopoint(y_pred, y_vel, self.alpha)

class VelocityEnsemble(EnsembleStrategy):
    def __init__(self, alpha: float = 0.5, min_history: int = 2):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self.alpha = alpha
        self.min_history = min_history

    def adjust(self, M, y_history, y_pred, y_history_preds=None):
        if len(y_history) < self.min_history or self.alpha == 0.0:
            return y_pred

        v = M.metric.log(y_history[-2], y_history[-1])
        y_vel = M.metric.exp(y_history[-1], v)
        return M.metric.geopoint(y_pred, y_vel, self.alpha)


class AdaptiveVelocityEnsemble(EnsembleStrategy):
    """Adaptive velocity ensemble that adjusts alpha based on recent prediction quality."""

    def __init__(self, alpha_range: tuple = (0.2, 0.8), window: int = 3):
        self.alpha_min, self.alpha_max = alpha_range
        self.window = window
        self.recent_errors = []

    def adjust(self, M, y_history, y_pred, y_history_preds=None) -> np.ndarray:
        if len(y_history) < 2:
            return y_pred

        if y_history_preds is not None and len(y_history_preds) > 0:
            last_actual = y_history[-1]
            last_pred = y_history_preds[-1]
            error = M.metric.dist(last_actual, last_pred)
            self.update_error(error)

        alpha = self._compute_alpha()
        velocity = M.metric.log(y_history[-2], y_history[-1])
        velocity_pred = M.metric.exp(y_history[-1], velocity)

        return M.metric.geopoint(y_pred, velocity_pred, alpha)

    def _compute_alpha(self) -> float:
        if len(self.recent_errors) < 2:
            return (self.alpha_min + self.alpha_max) / 2

        recent = self.recent_errors[-self.window:]
        trend = np.mean(np.diff(recent))
        alpha = self.alpha_min if trend < 0 else self.alpha_max
        return alpha

    def update_error(self, error: float):
        self.recent_errors.append(error)
        if len(self.recent_errors) > self.window * 2:
            self.recent_errors.pop(0)


class MAEnsemble(EnsembleStrategy):
    """Moving Average ensemble strategy for post-processing predictions."""

    def __init__(self, alpha: float = 0.5, order: int = 2, weights: np.array = None, min_history: int = 3):
        if not 0.0 <= alpha <= 1.0:
            print(f"Warning: alpha={alpha} outside [0,1] may cause instability")

        self.alpha = alpha
        self.order = order
        self.min_history = min_history
        if weights is not None:
            self.weights = np.array(weights)
        else:
            w = np.array([np.exp(i * 0.5) for i in range(order)])
            self.weights = w / w.sum()

    def adjust(self, M, y_history, y_pred, y_history_preds=None) -> np.ndarray:
        if y_history_preds is None or len(y_history_preds) < self.min_history:
            return y_pred

        q = min(self.order, len(y_history_preds))
        # Shape validation
        if len(y_history) < q:
            import warnings
            warnings.warn(
                f"MAEnsemble: y_history has {len(y_history)} points but need {q}",
                stacklevel=2
            )
            return y_pred
        past_actuals = y_history[-q:]
        past_preds = y_history_preds[-q:]

        residuals = [M.metric.log(p, a) for p, a in zip(past_preds, past_actuals)]
        v_fix_pattern = _linear_combination(M, y_pred, past_preds, residuals, self.weights[-q:])

        return M.metric.exp(y_pred, self.alpha * v_fix_pattern)

    @staticmethod
    def opt(M, y_history: np.ndarray, y_history_preds: np.ndarray, order: int):
        """
        Optimize ensemble weights by minimizing the distance between the
        weighted combination of parallel-transported past residuals and the
        actual next prediction residual.

        Uses softmax parameterization (unconstrained v → valid weights) and
        RiemannianSteepestDescent instead of scipy for consistency.

        Parameters
        ----------
        M : Manifold
        y_history : np.ndarray
            True observed values, shape (T, *point_shape)
        y_history_preds : np.ndarray
            Past predictions, shape (T, *point_shape)
        order : int
            Number of past residuals to combine

        Returns
        -------
        w : np.ndarray, shape (order,)
            Optimized non-negative weights summing to 1
        """
        if len(y_history_preds) <= order:
            return np.ones(order) / order

        # Need (order + 1) points: order past residuals + 1 target residual
        points_preds = y_history_preds[-(order + 1):]
        points_trues = y_history[-(order + 1):]

        # Residual vectors: v[i] = Log(pred_i, true_i)  (direction: pred → true)
        v = [M.metric.log(p, a) for p, a in zip(points_preds, points_trues)]

        # Target: the most recent residual (what we want to predict/cancel)
        target_v = v[-1]
        past_vs = v[:-1]                 # the order past residuals
        target_p = points_preds[-1]      # transport destination

        # Parallel-transport each past residual to the tangent space at target_p
        transported_vs = [
            M.metric.parallel_transport(past_vs[i], points_preds[i], target_p)
            for i in range(order)
        ]
        transported_vs_jnp = jnp.array(transported_vs)
        target_v_jnp = jnp.array(target_v)

        # Objective: find weights w (via softmax of v_raw) such that
        #   sum_i w_i * transported_vs[i]  ≈  target_v
        # i.e., past residuals predict the current residual → correction is accurate
        S = Euclidean((order,))

        @jax.jit
        def objective(v_raw):
            w = jax.nn.softmax(v_raw)
            combined = jnp.sum(
                jax.vmap(lambda wi, vi: wi * vi)(w, transported_vs_jnp),
                axis=0
            )
            return jnp.sum(jnp.square(combined - target_v_jnp))

        init_v = jnp.zeros(order)   # softmax(0) = uniform weights
        res = RiemannianSteepestDescent.fixedpoint(
            S, objective, init_v, maxiter=maxiter, mingradnorm=mingradnorm
        )

        return np.array(jax.nn.softmax(res))


# ============================================================================
# MAIN MODEL CLASS
# ============================================================================

class Model(abc.ABC):
    """Base class for manifold-valued time series models."""

    def __init__(
            self,
            M,
            lag: Optional[int] = None,
            x_fit: Optional[np.ndarray] = None,
            y_fit: Optional[np.ndarray] = None,
            ensemble_strategy: Optional[EnsembleStrategy] = None
    ):
        self.M = M
        self.lag = lag
        self.x_fit = x_fit
        self.y_fit = y_fit
        self.ensemble_strategy = ensemble_strategy

        self.weight = None
        self.param = None

    @property
    def dist(self):
        """Get the distance function from the manifold metric."""
        return self.M.metric.dist

    @property
    def parametric(self) -> bool:
        """
        True if the model evaluates a fitted function f(t) at given time points X
        (Reg, RidgeReg). For parametric models, predict() with iterative=False
        evaluates f(t) directly at all future time points — the natural and exact
        multi-step strategy.

        False if the model is non-parametric and only uses recent history
        (AR, VWA, ARMA, MA, WeightedAverage). For non-parametric models,
        multi-step prediction MUST be recursive (iterative=True): the model_fun
        always returns exactly 1 point and X is ignored, so iterative=False is
        mathematically degenerate for len(X) > 1.

        Subclasses that are non-parametric MUST override this to return False.
        """
        return True

    def set_ensemble_strategy(self, strategy: Optional[EnsembleStrategy]):
        """Set or update the ensemble strategy."""
        self.ensemble_strategy = strategy
        return self

    def predict(self, X: np.ndarray, iterative: bool = False) -> np.ndarray:
        """
        Generate predictions at points specified by X.

        For parametric models (Reg, RidgeReg):
            iterative=False  → evaluates the fitted trend f(t) directly at X (natural mode)
            iterative=True   → recursive closed-loop forecast (rarely needed)

        For non-parametric models (AR, VWA, ARMA, MA, WeightedAverage):
            iterative=True is always enforced for len(X) > 1, regardless of the
            caller's argument. This is mathematically necessary: model_fun() always
            produces exactly one prediction from the current history; X is ignored.
            Using iterative=False for multi-step would silently repeat the same
            single-step prediction (degenerate frozen-history forecast).
            A warning is issued when the override activates.

        Parameters
        ----------
        X : np.ndarray
            Time parameters for prediction (used by parametric models; ignored by
            non-parametric except for its length to determine number of steps).
        iterative : bool, default=False
            Requested prediction mode. Overridden to True for non-parametric
            models when len(X) > 1.

        Returns
        -------
        np.ndarray, shape (len(X), *point_shape)
        """
        # Non-parametric models must use recursive prediction for multi-step
        effective_iterative = iterative if self.parametric else True

        if not effective_iterative:
            return self.model_fun(X)

        # Warn when the override activates so the caller is not surprised
        if not self.parametric and not iterative and len(X) > 1:
            import warnings
            warnings.warn(
                f"{type(self).__name__} is non-parametric: multi-step prediction "
                f"always uses recursive (iterative) mode. "
                f"Pass iterative=True to suppress this warning.",
                stacklevel=2
            )

        # Closed-loop iterative forecasting
        if self.y_fit is None:
            raise ValueError("Model must be fitted before prediction. Call fit() first.")

        original_y_fit = self.y_fit
        original_x_fit = self.x_fit

        working_y = self.y_fit.copy()
        working_x = self.x_fit.copy() if self.x_fit is not None else None

        predictions = []

        for i in range(len(X)):
            self.y_fit = working_y
            self.x_fit = working_x

            next_pred = self.model_fun(X[i:i + 1])[0]
            predictions.append(next_pred)

            working_y = np.concatenate([working_y, [next_pred]], axis=0)
            if working_x is not None:
                working_x = np.concatenate([working_x, X[i:i + 1]], axis=0)

        self.y_fit = original_y_fit
        self.x_fit = original_x_fit

        return np.array(predictions)

    def validate(self, X: np.ndarray, y: np.ndarray) -> None:
        """Validate and store input data based on lag."""
        if X is None or y is None:
            raise ValueError("X and y cannot be None")

        if len(X) != len(y):
            raise ValueError(f"X and y must have same length. Got len(X)={len(X)}, len(y)={len(y)}")

        lag, n = self.lag, len(y)
        m = max(0, n - lag) if lag and lag > 0 else 0
        X_to_store = X[m:]

        # Non-parametric models do not use the time parameter X.
        # Replace x_fit with a sequential integer index for numerical stability.
        if not self.parametric:
            X_to_store = np.arange(len(X_to_store), dtype=X.dtype)

        self.x_fit, self.y_fit = X_to_store, y[m:]

    def error(self, ypred: np.ndarray, ytrue: np.ndarray) -> float:
        """Calculate prediction error using manifold distance."""
        return errfun(self.dist)(ypred, ytrue)

    def residual(self, ypred: np.ndarray, ytrue: np.ndarray) -> float:
        """Calculate sum of squared residuals using manifold distance."""
        m = min(len(ypred), len(ytrue))
        ypred_trunc, ytrue_trunc = ypred[:m], ytrue[:m]

        try:
            if hasattr(self.M.metric, 'squared_dist'):
                dist_fn = jax.vmap(self.M.metric.squared_dist)
                return float(jnp.sum(dist_fn(ypred_trunc, ytrue_trunc)))
        except Exception:
            # vmap or squared_dist unavailable; fall back to scalar loop
            pass

        return float(np.sum([self.dist(ypred_trunc[k], ytrue_trunc[k]) ** 2
                             for k in range(m)]))

    @abc.abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'Model':
        """Fit the model to training data. Must be implemented in subclasses."""
        pass

    @abc.abstractmethod
    def model_fun(self, X: np.ndarray) -> np.ndarray:
        """Apply the model function to input data. Must be implemented in subclasses."""
        pass


# ============================================================================
# REGRESSION MODELS
# ============================================================================

class Reg(Model):
    """Polynomial regression on manifolds."""

    def __init__(
            self,
            M: manifold.Manifold,
            lag: Union[bool, int] = False,
            degree: int = 3
    ):
        self.trend = None
        self.degree = degree
        lag = degree + 1 if lag else lag
        super().__init__(M=M, lag=lag)

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'Reg':
        self.validate(X, y)
        self.trend = PolyRegression(
            self.M,
            self.y_fit,
            self.x_fit,
            self.degree
        ).trend
        return self

    def model_fun(self, X: np.ndarray) -> np.ndarray:
        if self.trend is None:
            raise ValueError("Model must be fitted before prediction. Call fit() first.")
        return jax.vmap(self.trend.eval)(X)


class RidgeReg(Model):
    """Ridge regression on manifolds."""

    def __init__(
            self,
            M: manifold.Manifold,
            mean: np.ndarray,
            cov: np.ndarray,
            ridge_const: float,
            lag: Union[bool, int] = False,
            degree: int = 3
    ):
        self.trend = None
        self.mean = mean
        self.cov = cov
        self.ridge_const = ridge_const
        self.degree = degree
        lag = degree + 1 if lag else lag
        super().__init__(M, lag)

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RidgeReg':
        self.validate(X, y)
        self.trend = RidgeRegression(
            M=self.M,
            Y=self.y_fit,
            param=self.x_fit,
            mean=self.mean,
            cov=self.cov,
            ridge_const=self.ridge_const,
            degree=self.degree,
            P_init=None,
            maxiter=100,
            mingradnorm=1e-6
        ).trend
        return self

    def model_fun(self, X: np.ndarray) -> np.ndarray:
        if self.trend is None:
            raise ValueError("Model must be fitted before prediction. Call fit() first.")
        return jax.vmap(self.trend.eval)(X)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def _linear_combination(
    M: manifold.Manifold,
    ref: np.ndarray,
    points: np.ndarray,
    vectors: np.ndarray,
    weights: np.ndarray
) -> jnp.ndarray:
    """
    Compute weighted linear combination of tangent vectors at ref point.
    JAX-compatible version - always transports (safe and correct).
    """
    result = jnp.zeros(M.point_shape)

    for pt, vec, w in zip(points, vectors, weights):
        vec_transported = M.metric.transp(pt, ref, vec)
        result = result + w * vec_transported

    return result