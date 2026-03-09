# Importing Libraries and Packages
import numpy as np
import scipy.linalg
from numpy import linalg as lg
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, max_error
import jax
import jax.numpy as jnp
from morphomatics.manifold import Sphere, Manifold, Euclidean, PowerManifold, SPD
from morphomatics.opt import RiemannianSteepestDescent
from typing import Tuple, List
import jax.lax as lax
#from morphomatics.stats import RiemannianRegression
from morphomatics import manifold
from TimeSeries.reg import PolyRegression as PolyReg
from TimeSeries.bezier_polynom import BezierPolynom
from TimeSeries.opt import grad_desc, Stoch_GD
from scipy import optimize

S2 = Sphere()
dim = 3
N_SUBJ, N_SAMPLES = 218, 32
YEAR_SUBJ = [21, 41, 60, 75, 84, 96, 112, 130, 146, 166, 197, 218] #  util.get_subj_year(subj)[1:]
distance = lambda y, z: lg.norm(y - z)
# r2 = lambda y, y_pred: 1 - np.sum(distance(y, y_pred)**2)/np.sum(distance(y, np.mean(y))**2)
r2 = lambda y, y_pred: (r2_score(y, y_pred))
rel_err = lambda y, y_pred: lg.norm((y - y_pred)) / lg.norm(y)


def validate_pred(Y, Y_pred, n_learn=0, n_pred=None, show=True):
    n = len(Y)
    maxerr, rerr, mae = np.zeros(n), np.zeros(n), np.zeros(n)
    r2_val = np.zeros(n)
    for k in range(n):
        y, y_pred = Y[k], Y_pred[k]
        m = min(len(y), len(y_pred))
        if m > n_learn:
            y, y_pred = y[n_learn:m], y_pred[n_learn:m]
            diff = np.array([lg.norm(y[i] - y_pred[i]) for i in range(m)])
            rerr[k] = rel_err(y, y_pred)
            mae[k] = np.mean(diff)
            maxerr[k] = np.max(diff)  #max_error(y, y_pred)
            if m > 1:
                r2_val[k] = r2(y, y_pred)

    if show:
        if n_pred is None:
            print("Overall approximation:")
        #else:
            #print("{} hours Prediction: ".format(6*n_pred))
        print("mae: {:.4f}".format(mae.mean()))
        print("R^2 value: {:.4f}".format(r2_val.mean()))
        #print("relative error: {:.4f}".format(rerr.mean()))
        print("max error: {:.4f}".format(maxerr.mean()))
    return np.array(r2_val), rerr, mae


def svd_symmetric(A):
    [s, u] = lg.eig(A)  # eigenvalues and eigenvectors
    v = u.copy()
    v[:, s < 0] = -u[:, s < 0]  # replacing the corresponding columns with negative sign
    s = abs(s)
    return [u, s, v.T]


def load_hur(path, sph=True, wind=True):
    data = np.load(path, allow_pickle=True)
    subj, seq, ids, info = data['subj'], data['seq'], data['ids'], data['info']
    H = [seq[ids[i]:ids[i] + subj[i, 2], (3, 4, 5, 7, 8, 9, 2)] for i in range(N_SUBJ)]
    #if onlyhur is True:  # only hurricanes
    #    H = [H[i] for i in np.nonzero([0 if np.max(h[:, 2]) < 34 else 1 for h in H])[0]]
    a = (3, 4, 5) if sph is True else (0, 1)
    if wind == True: a = (*a, 2)
    H = [h[:, (*a, 6)] for h in H]
    return H


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


def cost_func(x, b, y, B):
    y_pred = eval_poly(x, b)  # y_pred = poly(x), b coefficients
    reg = np.sum(distance(y, y_pred) ** 2)
    mean_b, cov_b = np.mean(B, axis=0), np.cov(B.T)
    prior = mahal(b, B) ** 2  # np.sum(distance(b, B)**2)
    return reg + prior


def eval_x_poly(x, deg):
    return np.vander(x, deg + 1)


def eval_poly(x, b):
    # Eval predictor: polynomial --> Linear combination of standard basis
    # y = eval_x_poly(x, len(b)-1)@b  # np.array([np.sum(np.array([b[i]*x[j]**i for i in range(len(b))])) for j in range(len(x))])
    #return np.polyval(b, x)  # same
    return np.vander(x, len(b)) @ b


def pred_value_modify(y, y_train=None, Y_train=None, DTW=False):
    n_ini = len(y)
    if Y_train is None:
        y_learn = np.hstack((y, y_train[n_ini:]))
    else:
        N = len(Y_train)
        err = np.zeros(N)
        for k in range(N):
            err[k] = mean_squared_error(y, Y_train[k][:n_ini])
        ix = np.where(err == err.min())
        y_train = np.squeeze(Y_train[ix[0][0]])
        y_learn = np.hstack((y, y_train[n_ini:]))
    return y_learn


def pred_length_modify(y, y_train=None, Y_train=None, DTW=False):
    n_ini = len(y)
    if Y_train is None:
        y_learn = y_train
    else:
        N = len(Y_train)
        err = np.zeros(N)
        for k in range(N):
            err[k] = mean_squared_error(y, Y_train[k][:n_ini])
        ix = np.where(err == err.min())
        y_train = np.squeeze(Y_train[ix[0][0]])
        y_learn = np.hstack((y, y_train[n_ini:]))
    return len(y_learn)


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


def ridge_parameter(Y_true, mus, predfun, n_learn=1, n_pred=4):
    # Estimate ridge parameter
    n = len(mus)
    mse = np.zeros(n)
    for j in range(n):
        Y_pred = predfun(Y_true, n_learn=n_learn, n_pred=n_pred, mu=mus[j])[0]
        mse_j = []
        for k in range(len(Y_true)):
            y, y_pred = Y_true[k], Y_pred[k]
            rng = range(n_learn, min(len(y), len(y_pred)))
            if rng.stop - rng.start > 1:
                mse_j += [mean_squared_error(y[rng], y_pred[rng])]
        mse[j] = np.array(mse_j).mean()
    mu = mus[np.where(mse == mse.min())]
    return mu


def ridge_opt(Y, mus, predfun, n_learn=1, n_pred=4):
    # Estimate optimal ridge parameter
    n = len(mus)
    res = np.zeros(n)
    amae = np.zeros(n)
    for j in range(n):
        Y_pred_j, amae_j = predfun(Y, mus[j], n_learn=n_learn, n_pred=n_pred)
        amae[j] = np.nanmean(amae_j)
        #res[j] = np.mean([np.sum(np.square(Y_pred_j[k][n_pred-1::n_pred][:len(Y_true[k])-n_pred]-Y_true[k][n_pred:])) for k in range(len(Y_true))])
    mu = mus[np.where(amae == amae.min())]
    #mu = mus[np.where(res == res.min())]
    return mu

def opt_cov(Y, cov_inv, predfun, n_learn=1, n_pred=4):
    # Estimate optimal Tychonov matrix
    n_cp = len(cov_inv)
    def cost(S):
        S = np.reshape(S, (n_cp, n_cp))
        Ypred, amae = predfun(Y, n_learn=n_learn, n_pred=n_pred, mu=1, S=S*S.T)
        return np.nanmean(amae)
    S = RiemannianSteepestDescent.fixedpoint(Euclidean((n_cp**2,)), cost, init=np.reshape(cov_inv, -1))
    return S

def opt_grid(Y, predfun, rngs, D, U, n_learn, n_pred):
    #U, D, Vh = lg.svd(S, full_matrices=True)
    def cost(mueps):
        mu, eps = mueps
        Y_pred, amae = predfun(Y, n_learn=n_learn, n_pred=n_pred, mu=mu, S=U@np.diag(D+eps)@U.T)
        a, b =[y[n_pred:] for y in Y], [y[n_pred - 1::n_pred][:1-n_pred] for y in Y_pred]
        res = [np.dot(a[i]-b[i],a[i]-b[i]) for i in range(len(Y))]
        #return np.nanmean(amae)
        np.nanmean(res)
    resb = optimize.brute(cost, rngs, full_output=True, finish=None)
    return resb[0][0], resb[0][1], resb[1]

def opt_reg(Y, predfun, S, n_learn, n_pred):
    D, U = lg.eig(S)
    n_cp = len(D)
    D_ini = np.log(D)
    #S_ini = scipy.linalg.logm(S)
    S_ini = S
    #def f(d):
    def f(s):
        s = (s+s.T)/2
        #Y_pred, amae = predfun(Y, n_learn=n_learn, n_pred=n_pred, mu=1, S=U@np.diag(np.array(d))@U.T)
        Y_pred, amae = predfun(Y, n_learn=n_learn, n_pred=n_pred, mu=1, S=s)
        #a, b = [y[n_pred:] for y in Y], [y[n_pred - 1::n_pred][:1 - n_pred] for y in Y_pred]
        #res = [np.dot(a[i] - b[i], a[i] - b[i]) for i in range(len(Y))]
        l = np.nanmean(amae)
        #l = np.nanmean(res)
        return l
    def gradf(d):
        n_cp = len(d)
        a, l = np.eye(n_cp), np.zeros(n_cp)
        for i in range(n_cp):
            _, amae = predfun(Y, n_learn=n_learn, n_pred=n_pred, mu=1, S=U@np.diag(a[i])@U.T)
            l[i] = np.nanmean(amae)
        return l
    #p_out = grad_desc(M=Euclidean((n_cp,)), f=f, p_ini=D_ini)
    p_out = grad_desc(M=Euclidean((n_cp, n_cp)), f=f, p_ini=S_ini)
    #p_out = np.exp(p_out)
    #p_out = scipy.linalg.expm(p_out)
    #p_out = np.reshape(p_out, (n_cp, n_cp))
    p_out = (p_out + p_out.T) / 2
    return p_out

def opt_reg_SGD(Y, predfun, D, U, n_learn, n_pred):
    #U, D, _ = lg.svd(S, hermitian=True)
    n_cp = len(D)
    def f(d):
        _, amae = predfun(Y, n_learn=n_learn, n_pred=n_pred, mu=1, S=U@np.diag(np.array(d))@U.T)
        l = np.nanmean(amae)
        return l
    D = Stoch_GD(f, D)
    return D


def display_data(x, Y, n_pred=1, title=None, colors=None, legend=None):
    fig = plt.figure()
    plt.scatter(x, Y[0], s=20, marker='o', label='Data')
    plt.plot(x, Y[0], '-', c='b', linewidth=1)
    leg = ["Data"]
    if len(Y) > 1:
        #plt.plot(x, Y[1], c='gray', label='Fitted', linewidth=1)
        #leg += ["Fitted"]
        if len(Y) > 2:
            leg += ["Forecast"]
            #plt.plot(x[:n_learn], y[:n_learn], '--', c='orange', label='Estimated')
            #leg += ["Estimated"]
            plt.scatter(x[n_pred:], Y[2], s=15, c='red', marker='*', label='Forecast')
            plt.plot(x[n_pred:], Y[3], '--', c='red', linewidth=1)
    #plt.scatter(x, Y[0], s=10, label='Data')
    #plt.title("Test Data") if title is None else plt.title(title)
    plt.xlabel("Lifetime of hurricane (days)", fontsize=12)
    plt.ylabel("Intensity (knots)", fontsize=12)
    plt.legend(fontsize=12)
    plt.show()


def interpolate(f, n_points):
    """Interpolate a discrete curve with nb_points from a discrete curve
    Returns
    -------
    interpolation : discrete curve with n_points points
    """
    old_length = len(f)
    interpolation = np.zeros(n_points)
    no = (n_points - 1) / (old_length - 1)
    for j in range(old_length - 1):
        p = f[j]
        v = f[j + 1] - f[j]
        i = no * j
        while i < no * (j + 1):
            tij = i / no - j
            interpolation[int(i)] = p + tij * v
            i += 1
        interpolation[int(i)] = f[j + 1]
    return interpolation


def resample(subj, seq, ids, n_samples):
    subj_old, seq_old, ids_old = subj, seq, ids
    ids = np.zeros(N_SUBJ, int)
    f = np.zeros((N_SUBJ, n_samples))
    seq = np.zeros(n_samples * N_SUBJ)
    # upsample maxwinds via linear interpolation
    for i in range(N_SUBJ):
        a = seq_old[ids_old[i]:ids_old[i] + subj_old[i, 2]]
        f[i] = interpolate(np.reshape(a, (a.shape[0], 1)), n_samples).flatten()
        rg = range(i * n_samples, (i + 1) * n_samples)
        seq[rg] = f[i]
        subj[i, 2], ids[i] = n_samples, n_samples * i
    return subj, seq, ids


def create(deg=3):
    data = np.load('./datasets/hur.npz', allow_pickle=True)
    subj, seq, ids, info = data['subj'], data['seq'], data['ids'], data['info']
    H = [seq[ids[i]:ids[i] + subj[i, 2], (7, 8, 9)] for i in range(N_SUBJ)]
    W = [seq[ids[i]:ids[i] + subj[i, 2], 5] for i in range(N_SUBJ)]
    #x = None if n_samples is None else np.linspace(0.0, 1.0, n_samples)
    B_h, Y_h = fit_poly_dc(H, deg=deg)
    #B_w, Y_w = fit_poly(W, deg=deg)
    B_w, Y_w = fit_poly_dc(W, deg=deg)
    return B_h, B_w


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


def add_dmy(avg, y):
    y_tmp = np.vstack((y[0], avg(y[0], y[1])))
    for i in range(1, len(y) - 1):
        y_tmp = np.vstack((y_tmp, np.vstack((y[i], avg(y[i], y[i + 1])))))
    y_tmp = np.vstack((y_tmp, y[-1]))
    #y_tmp = np.array([[y[i], avg(y[i], y[i + 1])] for i in range(len(y) - 1)])
    #y_tmp = np.vstack((y_tmp.reshape((2*(len(y) - 1), -1)), y[-1]))
    return y_tmp

def avg_replace(avg, a, y, n):
    for j in range(n):
        #y = np.array(y)
        y[0] = avg(a, y[0])
        for i in range(1, len(y)):
            y[i] = avg(y[i], y[i - 1])
    return y

def amae_nhc(H_test, Y_pred, n_pred=4, dist=distance):
    amae = []
    for k in range(len(H_test)):
        h, p = H_test[k][n_pred:], Y_pred[k]
        ix = np.where(h[:, -1] > 0)[0]
        #ix = ix[np.remainder(ix[0], n_pred):]
        for i in ix[:-n_pred]:  #ix:
            amae += [dist(h[i, :-1], p[i])]
    return amae

def diff(M: manifold.Manifold, y, ref=None):
    if ref is None:
        ref = np.array([y[k] for k in range(len(y)-1)])
    return np.array([M.metric.log(ref[k], y[k+1]) for k in range(len(y)-1)])

def bez_sph(n_points):

    # 1. Define Control Points and Parameters
    P = np.array([
        [-4, 0, -1],
        [2, 1, 1],
        [-2, 0, -1],
        [0, 1, -1]
    ])
    t_values = np.linspace(0, 1, n_points + 1)
    deg = len(P) - 1

    # 2. Normalize Control Points to unit vectors (on the unit sphere)
    # This forces the points P_i to be "spherical."
    P_normalized = P / np.linalg.norm(P, axis=1, keepdims=True)

    # 3. Define the SLERP (Spherical Linear Interpolation) function
    def slerp(v0, v1, t):
        """
        Computes the Spherical Linear Interpolation between two unit vectors.
        """
        # Angle between v0 and v1
        dot_product = np.dot(v0, v1)
        dot_product = np.clip(dot_product, -1.0, 1.0)
        omega = np.arccos(dot_product)

        if omega < 1e-6: # Handle near-collinear vectors
            return (1 - t) * v0 + t * v1

        sin_omega = np.sin(omega)

        # SLERP formula
        slerp_point = (np.sin((1 - t) * omega) / sin_omega) * v0 + \
                      (np.sin(t * omega) / sin_omega) * v1

        # Re-normalize for numerical stability
        return slerp_point / np.linalg.norm(slerp_point)

    # 4. Implement the Spherical De Casteljau (SLERP) Algorithm for the Spherical Bézier Curve
    def sph_bez_curve_pt(t, P):
        """
        Computes a point on the Spherical Bézier Curve of any degree using
        the De Casteljau algorithm with SLERP.
        """
        points = list(P)

        # Iterate for each level of the De Casteljau algorithm
        for j in range(1, deg + 1):
            new_points = []
            for i in range(deg - j + 1):
                # SLERP is used instead of linear interpolation
                b_j_i = slerp(points[i], points[i+1], t)
                new_points.append(b_j_i)
            points = new_points

        return points[0]

    # 5. Compute the Curve Points
    y = np.zeros((len(t_values), 3))
    for i, t in enumerate(t_values):
        y[i] = sph_bez_curve_pt(t, P_normalized)

    #visSphere([y, y], ['b', 'r'])
    return y

def coord_2D3D(lat, lon, h=0.0):
    """
    this function converts latitude,longitude and height above sea level
    to earthcentered xyx coordinates in wgs84, lat and lon in decimal degrees
    e.g. 52.724156(West and South are negative), heigth in meters
    for algoritm see https://en.wikipedia.org/wiki/Geographic_coordinate_conversion#From_geodetic_to_ECEF_coordinates
    for values of a and b see https://en.wikipedia.org/wiki/Earth_radius#Radius_of_curvature
    """
    #a = 1  # 6378137.0             #radius a of earth in meters cfr WGS84
    #b = 1  # 6356752.3             #radius b of earth in meters cfr WGS84
    #e2 = 1 - (b ** 2 / a ** 2)
    latr = np.pi*lat/180  # latitude in radians
    lonr = np.pi*lon/180  # longituede in radians
    #Nphi = a / sqrt(1 - e2 * sin(latr) ** 2)
    x = np.cos(latr) * np.cos(lonr)  # (Nphi + h) * cos(latr) * cos(lonr)
    y = np.cos(latr) * np.sin(lonr)  # (Nphi + h) * cos(latr) * sin(lonr)
    z = np.sin(latr)  # (b ** 2 / a ** 2 * Nphi + h) * sin(latr)
    return x, y, z

def coord_3D2D(xyz):
    x, y, z = xyz[0], xyz[1], xyz[2]
    lat = np.sign(z)*180*np.arctan(z/np.sqrt(x**2 + y**2))/np.pi
    lon = 180*np.arctan2(y, x)/np.pi # West is negative
    return lat, lon


def visSphere(points_list, color_list, size=20, nice=True):
    """
    Visualize groups of points on the 2D-sphere
    """
    import matplotlib.pyplot as plt
    _ = plt.figure(figsize=(size, size))
    ax = plt.subplot(111, projection="3d")
    #ax.set_aspect("auto")  # equal
    ax.set_box_aspect([1.0, 1.0, 1.0])
    # draw sphere
    if nice:
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = 1 * np.outer(np.cos(u), np.sin(v))
        y = 1 * np.outer(np.sin(u), np.sin(v))
        z = 1 * np.outer(np.ones(np.size(u)), np.cos(v))

        # Create a light blue sphere with shading
        ax.plot_surface(x, y, z, color='#E8F4F8', alpha=0.25,
                        linewidth=0, antialiased=True, shade=True,
                        lightsource=plt.matplotlib.colors.LightSource(azdeg=45, altdeg=45))

        # Remove axes for cleaner look
        ax.set_axis_off()
        # Remove grid
        ax.grid(False)
    else:
        u, v = np.meshgrid(np.linspace(0.0, 2 * np.pi, 40), np.linspace(0.0, np.pi, 20))
        x, y, z = np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v)
        ax.plot_wireframe(x, y, z, color="grey", alpha=0.2)
    #ax.set_axis_off() # Turn off the axis planes
    # Set viewing angle
    #ax.view_init(elev=20, azim=45)
    # ax.set_title("")
    for i in range(len(points_list)):
        for points in points_list[i]:
            points = np.array(points)
            ax.scatter(points[0], points[1], points[2], s=10, color=color_list[i], marker=".")
    plt.show(block=True)

