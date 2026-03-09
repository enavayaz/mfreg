import abc
from typing import Optional, Tuple, Union
import numpy as np
import numpy.linalg as lg
from morphomatics.opt import RiemannianSteepestDescent
from morphomatics.manifold import Euclidean
from TimeSeries.verification_metrics import errfun


class LinearModel(abc.ABC):
    """
    Base class for linear time series models.

    Parameters
    ----------
    lag : int, optional
        Number of past samples to be used for prediction
    x_fit : np.ndarray, optional
        Fitted input values
    y_fit : np.ndarray, optional
        Fitted target values
    parametric : bool, optional
        Whether the model is parametric
    model_name : str, default="LinearModel"
        Name identifier for the model

    Attributes
    ----------
    coef : np.ndarray
        Model coefficients that define the linear model function.
        Evaluates x to return predicted y = coef @ transformed_x
    x_fit : np.ndarray
        Fitted input values
    y_fit : np.ndarray
        Fitted target values
    """

    def __init__(
            self,
            lag: Optional[int] = None,
            x_fit: Optional[np.ndarray] = None,
            y_fit: Optional[np.ndarray] = None,
            model_name: str = "LinearModel"
    ):
        self.lag = lag
        self.x_fit = x_fit
        self.y_fit = y_fit
        self._model_name = model_name
        self.coef = None

    def validate(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Validate and store input data based on lag.

        Parameters
        ----------
        X : np.ndarray
            Input features
        y : np.ndarray
            Target values

        Raises
        ------
        ValueError
            If X and y have incompatible shapes
        """
        if len(X) != len(y):
            raise ValueError(f"X and y must have same length. Got len(X)={len(X)}, len(y)={len(y)}")

        if X is None or y is None:
            raise ValueError("X and y cannot be None")

        lag, n = self.lag, len(y)
        m = max(0, n - lag) if lag and lag > 0 else 0
        self.x_fit, self.y_fit = X[m:], y[m:]

    @abc.abstractmethod
    def transf(self, X: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Transform input features. Must be implemented in subclasses.

        Parameters
        ----------
        X : np.ndarray, optional
            Input features to transform

        Returns
        -------
        np.ndarray
            Transformed features
        """
        pass

    def model_fun(self, X: np.ndarray) -> np.ndarray:
        """
        Apply the model function to input data.

        Parameters
        ----------
        X : np.ndarray
            Input features

        Returns
        -------
        np.ndarray
            Predicted values
        """
        if self.coef is None:
            raise ValueError("Model must be fitted before prediction. Call fit() first.")
        return self.transf(X) @ self.coef

    def predict(self, X: np.ndarray, iterative: bool = True) -> np.ndarray:
        """
        Predict target values for input data.

        Parameters
        ----------
        X : np.ndarray
            Input features
        iterative : bool, default=True
            Whether to use iterative prediction (refitting at each step)

        Returns
        -------
        np.ndarray
            Predicted values
        """

        if iterative:
            y = self.model_fun(X[:1])
            for i in range(len(X) - 1):
                x_fit = np.hstack((self.x_fit, X[i]))
                y_fit = np.append(self.y_fit, y)
                self.fit(x_fit, y_fit)
                y = np.vstack((y, self.model_fun(X[i + 1:i + 2])))
        else:
            y = self.model_fun(X)
        return y

    def error(self, ypred: np.ndarray, ytrue: np.ndarray) -> float:
        """
        Calculate prediction error using L2 norm.

        Parameters
        ----------
        ypred : np.ndarray
            Predicted values
        ytrue : np.ndarray
            True values

        Returns
        -------
        float
            Error value
        """
        return errfun(lambda a, b: lg.norm(a - b))(ypred, ytrue)

    def residual(self, ypred: np.ndarray, ytrue: np.ndarray) -> float:
        """
        Calculate sum of squared residuals.

        Parameters
        ----------
        ypred : np.ndarray
            Predicted values
        ytrue : np.ndarray
            True values

        Returns
        -------
        float
            Sum of squared residuals
        """
        m = min(len(ypred), len(ytrue))
        return np.sum((ypred[:m] - ytrue[:m]) ** 2)

    @abc.abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearModel':
        """
        Fit the model to training data. Must be implemented in subclasses.

        Parameters
        ----------
        X : np.ndarray
            Training features
        y : np.ndarray
            Training targets

        Returns
        -------
        LinearModel
            Fitted model instance
        """
        pass


class Reg(LinearModel):
    """
    Polynomial regression model using ordinary least squares.

    Fits a polynomial of specified degree to the data using least squares.

    Parameters
    ----------
    lag : bool or int, default=False
        If True, sets lag to deg+1. If int, uses that value.
    degree : int, default=3
        Degree of the fitted polynomial

    Attributes
    ----------
    coef : np.ndarray
        Polynomial coefficients
    rank_ : int
        Rank of the design matrix (to be implemented)
    singular_ : np.ndarray
        Singular values of the design matrix (to be implemented)

    Examples
    --------
    >>> model = Reg(degree=3)
    >>> X = np.array([0, 1, 2, 3])
    >>> y = np.array([1, 2, 5, 10])
    >>> model.fit(X, y)
    >>> predictions = model.predict(np.array([4, 5]))
    """

    def __init__(self, lag: Union[bool, int] = False, degree: int = 3):
        self.degree = degree
        lag = degree + 1 if lag else lag
        super().__init__(lag, parametric=True, model_name="PolyReg")

    def transf(self, X: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Transform input to polynomial features.

        Parameters
        ----------
        X : np.ndarray
            Input features

        Returns
        -------
        np.ndarray
            Vandermonde matrix of polynomial features
        """
        return poly_x(X, self.degree + 1)

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'Reg':
        """
        Fit polynomial regression model.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples,)
            Training data
        y : np.ndarray of shape (n_samples,)
            Target values

        Returns
        -------
        Reg
            Fitted model instance
        """
        self.validate(X, y)
        X, y = self.x_fit, self.y_fit
        A = self.transf(X)
        self.coef = lg.pinv(A.T @ A) @ (A.T @ y)
        return self


class RidgeReg(LinearModel):
    """
    Ridge polynomial regression with Tikhonov regularization.

    Fits a polynomial using least squares with Tikhonov regularization
    based on a prior mean and covariance.

    Parameters
    ----------
    mean : np.ndarray
        Prior mean for coefficients
    cov : np.ndarray
        Prior covariance matrix for coefficients
    ridge_const : float
        Ridge regularization constant
    lag : bool or int, default=False
        If True, sets lag to deg+1. If int, uses that value.
    degree : int, default=3
        Degree of the fitted polynomial

    Attributes
    ----------
    coef : np.ndarray
        Polynomial coefficients
    rank_ : int
        Rank of the design matrix (to be implemented)
    singular_ : np.ndarray
        Singular values of the design matrix (to be implemented)
    """

    def __init__(
            self,
            mean: np.ndarray,
            cov: np.ndarray,
            ridge_const: float,
            lag: Union[bool, int] = False,
            degree: int = 3
    ):
        self.mean = mean
        self.cov = cov
        self.ridge_const = ridge_const
        self.degree = degree
        lag = degree + 1 if lag else lag
        super().__init__(lag, parametric=True, model_name="RidgePolyReg")

    def transf(self, X: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Transform input to polynomial features.

        Parameters
        ----------
        X : np.ndarray
            Input features

        Returns
        -------
        np.ndarray
            Vandermonde matrix of polynomial features
        """
        return poly_x(X, self.degree + 1)

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RidgeReg':
        """
        Fit ridge regression model.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples,)
            Training data
        y : np.ndarray of shape (n_samples,)
            Target values

        Returns
        -------
        RidgeReg
            Fitted model instance

        Raises
        ------
        ValueError
            If covariance matrix is singular
        """
        self.validate(X, y)
        X, y = self.x_fit, self.y_fit

        try:
            S = self.ridge_const * lg.inv(self.cov)
        except lg.LinAlgError:
            raise ValueError("Covariance matrix is singular and cannot be inverted")

        A = self.transf(X)
        self.coef = lg.pinv(A.T @ A + S) @ (A.T @ y + S @ self.mean)
        return self


class AR(LinearModel):
    """
    Autoregressive (AR) model for Euclidean space.

    Predicts the next point by applying a weighted linear combination of
    past velocities (differences) to the current position.
    """

    def __init__(
            self,
            lag: Union[bool, int] = True,
            order: int = 2,
            warm_start: bool = True
    ):
        self.order = order
        # Consistent with manifold AR: lag is 2*order + 1 if True
        lag = 2 * order + 1 if lag is True else lag
        self.warm_start = warm_start
        self._prev_weight: Optional[np.ndarray] = None
        super().__init__(lag=lag, model_name="AR")

    def transf(self, X: Optional[np.ndarray] = None) -> np.ndarray:
        """Required by LinearModel base class; returns the most recent point."""
        return self.y_fit[-1:]

    def compute_par(self) -> np.ndarray:
        """
        Optimizes weights w using RiemannianSteepestDescent on a
        Euclidean manifold.
        """
        y = self.y_fit
        n = len(y)
        p = self.order

        if n < p + 1:
            return np.ones(p) / p

        # Precompute 'velocities' (differences)
        # dy[i] = y[i+1] - y[i]
        dy = np.diff(y, axis=0)

        def cost(w: np.ndarray) -> float:
            """
            Squared error cost function.
            Weights are normalized via softmax to preserve momentum.
            """
            # Softmax normalization for weight stability
            w_norm = np.exp(w) / np.sum(np.exp(w))

            s = 0.0
            # Start loop when we have enough history to look back 'p' steps
            for k in range(p, n):
                ref_k = y[k - 1]

                # Weighted combination of past differences
                v_ar = np.zeros_like(y[0])
                for i in range(p):
                    # dy[k-p+i-1] matches the transition associated with y[k-p+i]
                    v_ar += w_norm[i] * dy[k - p + i - 1]

                # Linear prediction: y_pred = y[k-1] + v_ar
                y_pred = ref_k + v_ar
                s += np.sum((y[k] - y_pred) ** 2)
            return float(s)

        # Optimization on a Euclidean manifold of dimension 'p'
        N = Euclidean((p,))

        # Initialization
        if self.warm_start and self._prev_weight is not None:
            w_init = self._prev_weight
        else:
            w_init = np.zeros(p)  # Log-space zeros = uniform weights after softmax

        w_opt = RiemannianSteepestDescent.fixedpoint(
            N, cost, w_init, maxiter=80, mingradnorm=1e-5
        )

        self._prev_weight = w_opt
        return w_opt

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'AR':
        """Fit the AR model by optimizing weights."""
        self.validate(X, y)
        self.coef = self.compute_par()
        return self

    def model_fun(self, X: np.ndarray) -> np.ndarray:
        """
        Applies the fitted model for a single-step prediction.
        """
        if self.coef is None:
            raise ValueError("Model must be fitted before prediction.")

        p = self.order
        y = self.y_fit
        ref = y[-1]

        # Normalize optimized coefficients
        w_norm = np.exp(self.coef) / np.sum(np.exp(self.coef))

        # Calculate recent velocities
        dy_recent = np.diff(y[-(p + 1):], axis=0)

        v_ar = np.zeros_like(y[0])
        for i in range(p):
            v_ar += w_norm[i] * dy_recent[i]

        pred = ref + v_ar
        return np.expand_dims(pred, axis=0)


class ARMA(LinearModel):
    """
    ARMA model for Euclidean space using RiemannianSteepestDescent.
    Matches the structure of the manifold-valued ARMA while using
    standard vector arithmetic.
    """

    def __init__(self, order: Tuple[int, int] = (2, 1), lag: Union[bool, int] = True):
        self.order = order
        p, q = order
        lag = 2 * max(p, q) + 1 if lag is True else lag
        super().__init__(lag=lag, model_name="ARMA")
        self._prev_weight: Optional[np.ndarray] = None

    def transf(self, X: Optional[np.ndarray] = None) -> np.ndarray:
        """Identity transformation required by base class."""
        return self.y_fit[-1:]

    def compute_par(self) -> np.ndarray:
        """Optimizes weights using RiemannianSteepestDescent on a Euclidean manifold."""
        y = self.y_fit
        n = len(y)
        p, q = self.order

        # dy[i] is the velocity from y[i] to y[i+1]
        dy = np.diff(y, axis=0)

        def cost(w: np.ndarray) -> float:
            # Decoupled weights: Softmax for AR, independent for MA
            w_ar = np.exp(w[:p]) / np.sum(np.exp(w[:p]))
            w_ma = w[p:]

            s = 0.0
            # Initialize residuals as zero vectors
            residuals = [np.zeros_like(y[0]) for _ in range(q)]

            for k in range(max(p, q), n):
                ref_k = y[k - 1]

                # AR Component: Weighted combination of past differences
                v_ar = np.zeros_like(y[0])
                for i in range(p):
                    v_ar += w_ar[i] * dy[k - p + i - 1]

                # MA Component: Weighted combination of past residuals
                v_ma = np.zeros_like(y[0])
                for i in range(q):
                    v_ma += w_ma[i] * residuals[-(i + 1)]

                # Prediction: y_pred = y[k-1] + v_ar + v_ma
                y_pred = ref_k + v_ar + v_ma

                # L2 Squared Error
                error = y[k] - y_pred
                s += np.sum(error ** 2)

                # Update residual buffer
                if q > 0:
                    residuals.append(error)
                    if len(residuals) > q:
                        residuals.pop(0)
            return float(s)

        # Weight space is a Euclidean manifold of dimension p + q
        N = Euclidean((p + q,))

        # Initialize AR as uniform and MA at zero
        w_init = self._prev_weight if self._prev_weight is not None else np.zeros(p + q)

        # Optimize using RiemannianSteepestDescent
        w_opt = RiemannianSteepestDescent.fixedpoint(
            N, cost, w_init, maxiter=80, mingradnorm=1e-5
        )

        self._prev_weight = w_opt
        return w_opt

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ARMA':
        self.validate(X, y)
        self.coef = self.compute_par()
        return self

    def model_fun(self, X: np.ndarray) -> np.ndarray:
        """One-step prediction for forecasting."""
        if self.coef is None:
            raise ValueError("Model must be fitted before prediction.")

        p, q = self.order
        y = self.y_fit
        ref = y[-1]

        w_ar = np.exp(self.coef[:p]) / np.sum(np.exp(self.coef[:p]))

        # AR Momentum using last p differences
        dy_recent = np.diff(y[-(p + 1):], axis=0)
        v_ar = np.zeros_like(y[0])
        for i in range(p):
            v_ar += w_ar[i] * dy_recent[i]

        # Standard ARMA forecasting assumes zero future residuals
        pred = ref + v_ar
        return np.expand_dims(pred, axis=0)


class WeightedAverage(LinearModel):
    """
    Weighted average model for Euclidean space.

    Predicts the next point as a convex combination of the 'lag' most
    recent points in the history.
    """

    def __init__(self, lag: int):
        # lag here determines how many past points to average
        super().__init__(lag=lag, model_name="WeightedAverage")
        self._prev_weight: Optional[np.ndarray] = None

    def transf(self, X: Optional[np.ndarray] = None) -> np.ndarray:
        """Identity transformation required by LinearModel base class."""
        return self.y_fit[-1:]

    def compute_par(self) -> np.ndarray:
        """
        Optimizes weights w using RiemannianSteepestDescent.
        The weights define a convex combination of historical points.
        """
        y = self.y_fit
        n = len(y)
        p = self.lag

        if n <= p:
            # Not enough history to fit weights, return uniform
            return np.zeros(p)

        def cost(w: np.ndarray) -> float:
            """
            Squared error cost function.
            Weights are softmax-normalized to ensure a convex combination.
            """
            # Ensure weights sum to 1 and are positive
            w_norm = np.exp(w) / np.sum(np.exp(w))

            s = 0.0
            # We slide a window of size 'p' to predict the next point 'k'
            for k in range(p, n):
                # Points used for the average: y[k-p], ..., y[k-1]
                history_window = y[k - p:k]

                # Convex combination (Weighted Average)
                y_pred = np.zeros_like(y[0])
                for i in range(p):
                    y_pred += w_norm[i] * history_window[i]

                s += np.sum((y[k] - y_pred) ** 2)
            return float(s)

        # Optimization on a Euclidean manifold of dimension 'p' (the lag)
        N = Euclidean((p,))

        # Initialization (uniform weights in log-space)
        w_init = self._prev_weight if self._prev_weight is not None else np.zeros(p)

        w_opt = RiemannianSteepestDescent.fixedpoint(
            N, cost, w_init, maxiter=80, mingradnorm=1e-5
        )

        self._prev_weight = w_opt
        return w_opt

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'WeightedAverage':
        """Fit the weighted average by optimizing historical weights."""
        self.validate(X, y)
        self.coef = self.compute_par()
        return self

    def model_fun(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the next point as the weighted average of the last 'lag' points.
        """
        if self.coef is None:
            raise ValueError("Model must be fitted before prediction.")

        p = self.lag
        y = self.y_fit

        # Normalize optimized coefficients
        w_norm = np.exp(self.coef) / np.sum(np.exp(self.coef))

        # Most recent window of size 'p'
        history_window = y[-p:]

        pred = np.zeros_like(y[0])
        for i in range(p):
            pred += w_norm[i] * history_window[i]

        return np.expand_dims(pred, axis=0)

# Utility functions

def poly_x(x: np.ndarray, length: int) -> np.ndarray:
    """
    Generate Vandermonde matrix for polynomial features.

    Parameters
    ----------
    x : np.ndarray
        Input values
    length : int
        Number of polynomial terms (degree + 1)

    Returns
    -------
    np.ndarray
        Vandermonde matrix of shape (len(x), length)
    """
    return np.vander(x, length)