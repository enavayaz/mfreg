import numpy as np
import numpy.linalg as lg
import jax
import jax.numpy as jnp
#from sklearn.model_selection import train_test_split
from sklearn.covariance import empirical_covariance, MinCovDet
from util_pred import cov_mat, eval_poly_hur, fit_poly_dc, N_SUBJ, ridge_parameter, ridge_opt
from util import visSphere, coord_2D3D
from morphomatics.manifold import Sphere, PowerManifold
from morphomatics.stats import ExponentialBarycenter, RiemannianRegression
from TimeSeries.verification_metrics import errfun
from TimeSeries.opt import *
from TimeSeries.stats import PrincipalGeodesicAnalysis as PGA
from TimeSeries.model import RidgeReg as RidgeModel

# Set random seed for reproducibility (optional)
np.random.seed(42)

# Global constants and parameters
n_subj, n_points = 50, 50
lon_max = .75*np.pi
lat_max = np.pi/20  # 4.5
deg, n_learn, n_pred = 5, 1, 2
n_cp = deg + 1
M = Sphere()
P = PowerManifold(M, n_cp)
dist = jax.jit(M.metric.dist)
err = errfun(M.metric.dist)
def map2D3D(x, y, uniform=True):
    #Z = np.zeros((n_points, 3))
    if uniform:
        Z = np.sqrt(1-y**2)*np.cos(x), np.sqrt(1-y**2)*np.sin(x), y
    else:
        Z = np.cos(y)*np.cos(x), np.cos(y)*np.sin(x), np.sin(y)  # central distribution
    return Z
#map2D3D = jax.vmap(map2D3D)

# Generate list of random trajectories
def rand_trjs(n_mat=n_subj, n_points=n_points, uniform=True):
    Y = []
    for i in range(n_mat):
        # Generate x coordinates: random but sorted (increasing)
        x = np.sort(np.random.uniform(0, lon_max, n_points))

        # Generate y coordinates: random
        if uniform:
            y_max = np.sin(lat_max)
            y = np.random.uniform(-y_max, y_max, n_points)
        else:
            y = np.random.uniform(-lat_max, lat_max, n_points)
        y = np.sort(y)
        # map to sphere
        z = map2D3D(x, y, uniform=uniform)
        Y.append(np.array([z[0], z[1], z[2]]).T)
    return Y

def gauss_trjs(n_mat=n_subj, n_points=n_points):
    noise_std = np.pi / 20  # Standard deviation for Gaussian noise

    # Generate mean trajectory on sphere directly
    x = np.sort(np.random.uniform(0.0, lon_max, n_points))
    y_max = np.sin(lat_max)
    y = np.sort(np.random.uniform(0.0, y_max, n_points))

    # Convert mean trajectory to 3D coordinates
    #mean_trj = np.zeros((n_points, 3))
    z = map2D3D(x, y, uniform=True)
    mean_trj = np.array([z[0], z[1], z[2]]).T

    # Generate n_mat trajectories by adding tangential noise to mean_trj
    Y = []
    for i in range(n_mat):
        noisy_trj = np.zeros((n_points, 3))
        for j in range(n_points):
            # Create tangent vectors perpendicular to mean point
            p = mean_trj[j]
            # Generate two orthogonal tangent vectors
            u = np.array([-p[1], p[0], 0])
            u = u / np.linalg.norm(u)
            v = np.cross(p, u)

            # Add random tangential noise
            noise_u = np.random.normal(0, noise_std)
            noise_v = np.random.normal(0, noise_std)

            # Combine noise vectors and project to sphere
            #point = p + noise_u * u + noise_v * v
            #noisy_trj[j] = point / np.linalg.norm(point)
            noisy_trj[j] = M.metric.exp(p, noise_u * u + noise_v * v)

        Y.append(noisy_trj)

    return Y

def save(Y, B):
    np.savez('../datasets/sphYB.npz', Y=Y, B=B)

def load():
    data = np.load('../datasets/sphYB.npz', allow_pickle=True)
    return data['Y'], data['B']

# Read Data
#Y = rand_trjs(n_mat=n_subj, n_points=n_points)
#Y = gauss_trjs(n_mat=n_subj, n_points=n_points)
##mean_len = int(np.mean([len(Y[i]) for i in range(len(Y))]))
#B, Y_fit = fit_poly_dc(M, Y, deg=deg)
#save(Y, B)
Y, B = load()
#Y = Y_fit
n_train = 40
Y_train, B_train, Y_test = Y[:n_train], B[:n_train], Y[n_train:]
n_test = len(Y_test)
# Covariance matrix and mean
Ex = ExponentialBarycenter()
mean_b_train = Ex.compute(P, B_train, max_iter=30)
cov_b_train = cov_mat(P.metric.log, B_train, mean_b_train) + 1e-6*np.eye(n_cp*dim)
cov_inv = lg.pinv(cov_b_train)

def pred(Y_test, mu):
    model = RidgeModel(M, mean_b_train, cov_inv, mu, lag=True, deg=deg)
    Y_pred, n_test = [], len(Y_test)
    # len_x = len_mean
    # x = np.linspace(0.0, 1.0, len_x)
    for k in range(n_test):
        y_test = Y_test[k]
        y_pred = []
        for n in range(n_learn, len(y_test)):
            y_learn = y_test[:n]
            if n > n_cp: y_learn = y_test[n - n_cp:n]
            len_x = n + n_pred
            x = np.linspace(0.0, 1.0, len_x)
            model = model.fit(x[:len(y_learn)], y_learn)
            #b = model.trend.control_points
            p = np.array(model.predict(x[n:len_x], iterative=False))
            p[-1] = M.metric.geopoint(y_learn[-1], p[-1], .5)
            y_pred += [p[0]]
        Y_pred += [np.array(y_pred)]
    return Y_pred

Ytest = [Y_test[0]]
mu = 1e-8
Y_pred = pred(Ytest, mu)
mae = [err(Y_pred[k], Ytest[k][1:]).mae() for k in range(len(Ytest))] #[err(Y_pred[k], Ytest[k][1:]).mae() for k in range(1)]
print('MAE: {:.4f}'.format(np.mean(mae)))
#r2_val, acc, mae = validate_pred(Y_test, Y_pred, n_learn=n_learn, n_pred=n_pred)

mus = np.logspace(-3, 2, 100)
#mu = ridge_parameter(Y_train, mus, predict, n_learn, n_pred)  # bad
mu = ridge_opt(Y_train, mus=mus, predfun=pred, n_learn=n_learn, n_pred=n_pred)
Y_pred = pred(Ytest, n_learn, n_pred)
print('MAE with opt mu: {:.4f}'.format(np.mean(mae)))
r2_val, acc, mae = validate_pred(Y_test, Y_pred, n_learn=n_learn, n_pred=n_pred)

k = 0
y, ypred, err = Ytest[k], Y_pred[k], mae[k]
x = np.linspace(0.0, 1.0, len(y))
ytest = fit_poly_dc (M, [y], deg=deg)
visSphere([y,ypred], ['r', 'b'])