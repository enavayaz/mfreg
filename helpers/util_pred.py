# Importing Libraries and Packages
import numpy as np
from numpy import linalg as lg
#from sklearn.metrics import mean_squared_error, r2_score
import jax
import jax.numpy as jnp
from morphomatics.manifold import Sphere, Manifold, Euclidean
from morphomatics.opt import RiemannianSteepestDescent
from typing import Tuple, List
import jax.lax as lax
from morphomatics import manifold
from timeseries.reg import PolyRegression as PolyReg
from timeseries.bezier_polynom import BezierPolynom


S2 = Sphere()
dim = 3
distance = lambda y, z: lg.norm(y - z)
#r2 = lambda y, y_pred: (r2_score(y, y_pred))
rel_err = lambda y, y_pred: lg.norm((y - y_pred)) / lg.norm(y)


def svd_symmetric(A):
    [s, u] = lg.eig(A)  # eigenvalues and eigenvectors
    v = u.copy()
    v[:, s < 0] = -u[:, s < 0]  # replacing the corresponding columns with negative sign
    s = abs(s)
    return [u, s, v.T]


def cov_mat(log, y, mean_y):
    n = len(y)
    w = jax.vmap(jax.jit(log), (None, 0))(mean_y, y)  # Map data to mean tangent space
    #w = [log(mean_y, y) for k in range(n)] #same
    w_vec = w.reshape(n, -1)
    return 1 / n * w_vec.T @ w_vec


def mahal(p, Q=None, mean=None, cov=None):
    if Q is None:
        u, cov = p - mean, cov
    else:
        u, cov = p - np.mean(Q, axis=0), np.cov(Q)

    # from scipy.spatial import distance
    # distance.mahalanobis(p, mean, lg.inv(np.cov(Q)))  # same
    return np.sqrt(u @ lg.inv(cov) @ u)


def eval_x_poly(x, deg):
    return np.vander(x, deg + 1)


def eval_poly(x, b):
    # Eval predictor: polynomial --> Linear combination of standard basis
    # y = eval_x_poly(x, len(b)-1)@b  # np.array([np.sum(np.array([b[i]*x[j]**i for i in range(len(b))])) for j in range(len(x))])
    #return np.polyval(b, x)  # same
    return np.vander(x, len(b)) @ b


def fit_poly1d(Y, deg, n_samples=None):
    """
    Fit polynomial of degree deg to data Y via least square regression
    :param Y: Values to fit
    :param deg: Degree of polynomial
    :return: Coefficients B_f of polynomials, fitted values Y_f,
    R2 values r2_val
    """
    N = len(Y)
    Y_f, B_f = [], []  #np.zeros((N, deg + 1))
    for k in range(N):
        y = Y[k]
        x = np.linspace(0, 1, len(y))
        b = np.polyfit(x, y, deg)
        B_f += [b]
        Y_f += [np.polyval(b, x)]
    #return B_f, Y_f
    return np.array(B_f), Y_f


def fit_poly(Y, deg, normalize=True):
    """
    Fit polynomial of degree deg to data Y via least square regression
    :param Y: Values to fit
    :param deg: Degree of polynomial
    :return: Coefficients B_f of polynomials, fitted values Y_f,
    R2 values r2_val
    """
    Y_f, B_f = [], []  #np.zeros((N, deg + 1, dim))
    for y in Y:
        len_y = 1.0 if normalize is True else len(y)
        x = np.linspace(0.0, len_y, len(y))
        #X = np.vander(x, deg + 1)
        #b = lg.inv(X.T@X)@X.T@y polyfit is more precise
        b = np.polyfit(x, y, deg)
        B_f += [b]
        Y_f += [np.vander(x, deg + 1) @ b]
    #return B_f, Y_f
    return np.array(B_f), Y_f


def fit_poly_dc(M: Manifold, trjs, deg=3, x=None):
    Coeff, Y = [], []
    for trj in trjs:
        trend = PolyReg(M, jnp.array(trj), jnp.linspace(0., 1., len(trj)), deg).trend
        Coeff += [np.array(trend.control_points)]
        #if x is None:
        x = np.linspace(0.0, 1.0, len(trj))
        #Y += [np.array([trend.eval(t) for t in x])]
        Y += [jax.vmap(trend.eval)(x)]
    return jnp.array(Coeff), Y


def opt_cov(Y, cov_inv, predfun, n_learn=1, n_pred=4):
    # Estimate optimal Tychonov matrix
    n_cp = len(cov_inv)
    def cost(S):
        S = np.reshape(S, (n_cp, n_cp))
        Ypred, amae = predfun(Y, n_learn=n_learn, n_pred=n_pred, mu=1, S=S*S.T)
        return np.nanmean(amae)
    S = RiemannianSteepestDescent.fixedpoint(Euclidean((n_cp**2,)), cost, init=np.reshape(cov_inv, -1))
    return S


eval_poly_hur = lambda b, x: jax.vmap(BezierPolynom(S2, b).eval)(x)


def eval_poly_dc(M: Manifold, P: jnp.array, x: jnp.array) -> jnp.array:
    """Evaluates the Bézier spline at time t."""
    ev = jax.jit(lambda t: decasteljau(M, P, t)[0])
    return jax.vmap(ev)(x)


def decasteljau(M: Manifold, P: jnp.array, t: float) -> Tuple[jnp.array, List[jnp.array]]:
    """Generalized de Casteljau algorithm
    :param M: manifold
    :param P: control points of curve beta
    :param t: scalar in [0,1]
    :return  beta(t), (B): result of the de Casteljau algorithm with control points P, (intermediate points Bf in the algorithm)
    """
    # number of control points
    k = len(P)

    # init linearized tree of control points
    B = jnp.concatenate([jnp.asarray(P)[i:] for i in range(k)])
    # for lower-level control points: indices of parent ones w.r.t Bf
    offset = [(2 * k * n - n * n + n) // 2 for n in range(k - 1)]
    idx = np.concatenate([np.arange(k - 1 - i) + o for i, o in enumerate(offset)])
    # compute lower-level points
    f = lambda B, io: (B.at[io[1]].set(M.connec.geopoint(B[io[0]], B[io[0] + 1], t)), None)
    B = lax.scan(f, B, np.c_[idx, k + np.arange(len(idx))])[0]

    return B[-1], [B[o:o + k - i] for i, o in enumerate(offset)]


avg = lambda x, y: (x + y) / 2


def avg_replace(avg, a, y, n):
    for j in range(n):
        #y = np.array(y)
        y[0] = avg(a, y[0])
        for i in range(1, len(y)):
            y[i] = avg(y[i], y[i - 1])
    return y


def diff(M: manifold.Manifold, y, ref=None):
    if ref is None:
        ref = np.array([y[k] for k in range(len(y)-1)])
    return np.array([M.metric.log(ref[k], y[k+1]) for k in range(len(y)-1)])

