from __future__ import annotations
import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
from typing import Callable
#import autograd as ag

ensemble = jax.jit(lambda M, p, q, y, alpha: M.metric.geopoint(y, M.metric.exp(q, -M.metric.log(q, p)), alpha))

#==========================================
# Steepet Descent
#==========================================

class RiemannianSteepestDescent(object):

    @staticmethod
    @partial(jax.jit, static_argnames=['f'])
    def fixedpoint(M, f: Callable[[jnp.array], float], init: jnp.array, params=(),
                   stepsize=1., maxiter=100, mingradnorm=1e-6) -> jnp.array:
        """
        Compute minimizer of f w.r.t. first argument.
        :param M: manifold search space
        :param f: objective function s.t. f(x, *params) -> float
        :param init: initial guess in M
        :param params: additional parameters for f
        :param stepsize: fixed length of step in steepest descent direction
        :param maxiter: maximum number of iterations in steepest descent
        :param mingradnorm: stop iteration when the norm of the gradient is lower than mingradnorm
        :return: Minimizer of f.
        """

        # Gradient
        grad_ = jax.grad(f)
        def grad(x):
            return M.metric.egrad2rgrad(x, grad_(x, *params))

        # optimize
        def body(args):
            x, _, i = args
            g = grad(x)
            g_norm = M.metric.norm(x, g)
            # steepest descent
            x = M.connec.exp(x, -stepsize * g)
            return x, g_norm, i + 1

        def cond(args):
            _, g_norm, i = args
            c = jnp.array([g_norm > mingradnorm, i < maxiter])
            return jnp.all(c)

        opt, *_ = jax.lax.while_loop(cond, body, (init, jnp.array(1.), jnp.array(0)))

        return opt

# ==========================================
# Stochastic Gradient Descent
# ==========================================

class SGD:
    def __init__(self, lr=0.01, epochs=1000, batch_size=32, tol=1e-3):
        self.learning_rate = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.tolerance = tol
        self.weights = None
        self.bias = None

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

    def mean_squared_error(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def gradient(self, X_batch, y_batch):
        y_pred = self.predict(X_batch)
        error = y_pred - y_batch
        gradient_weights = np.dot(X_batch.T, error) / X_batch.shape[0]
        gradient_bias = np.mean(error)
        return gradient_weights, gradient_bias

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.random.randn(n_features)
        self.bias = np.random.randn()

        for epoch in range(self.epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for i in range(0, n_samples, self.batch_size):
                X_batch = X_shuffled[i:i + self.batch_size]
                y_batch = y_shuffled[i:i + self.batch_size]

                gradient_weights, gradient_bias = self.gradient(X_batch, y_batch)
                self.weights -= self.learning_rate * gradient_weights
                self.bias -= self.learning_rate * gradient_bias

            if epoch % 100 == 0:
                y_pred = self.predict(X)
                loss = self.mean_squared_error(y, y_pred)
                print(f"Epoch {epoch}: Loss {loss}")

            if np.linalg.norm(gradient_weights) < self.tolerance:
                print("Convergence reached.")
                break

        return self.weights, self.bias


def grad_desc(M, f, p_ini, gradf=None, lrate=1., max_iter=20, tol=1e-2):
    """
    Apply a gradient descent until either max_iter or a given tolerance is reached.
    """
    n_cp = len(p_ini)
    if gradf is None:
        a, h = np.zeros_like(p_ini), .1*tol
        def gradf(p):
            egrad = np.zeros_like(a)
            for i, j in np.ndindex(np.shape(a)):
            #egrad = np.array([f(p+h*a[i])-f(p) for i in range(n_cp)])/h
            #egrad = .5*np.array([f(p+h*a[i, j])-f(p-h*a[i, j]) for i, j in np.ndindex(n_cp, n_cp)])/h
                a[i, j] = 1.0
                egrad[i, j] = f(p+h*a)-f(p-h*a)
            egrad = .5*egrad/h
            #for i in range(n_cp):
            #    for j in range(n_cp):
            #        a[i,j] = .5*(f(p+h*a[i,j]) - f(p-h*a[i,j]))/h
            #egrad = jax.grad(f)
            #egrad = ag.grad(f)
            return M.proj(p, egrad)
            #return M.metric.egrad2rgrad(p, egrad)
    p = p_ini
    for i in range(max_iter):
        grad_p = gradf(p)
        #grad_norm = np.linalg.norm(grad_p.flatten())
        grad_norm = M.metric.norm(p, grad_p)
        print('cost: {:.4f}, grad_norm: {:.4f}'.format(f(p), grad_norm))
        if grad_norm < tol:
            break
        p = M.metric.exp(p, -lrate * grad_p)
    return p


def Stoch_GD(f, x_ini, lr=0.1, epochs=1000, batch_size=32, tol=1e-3):
    # Create random dataset with n_sample rows and n_cp columns
    def f(x):
        return np.dot(x,x)+55
    x_ini = 0.0*x_ini
    n_cp = len(x_ini)
    n_samples = 10
    X = x_ini+np.random.randn(n_samples, n_cp)
    Xx = np.random.randn(10, 5)
    # create corresponding target value by adding random
    # noise in the dataset
    Y = np.array([f(X[i])+np.random.randn(1) * 0.1 for i in range(n_samples)]).ravel()
    Yy = np.dot(Xx, np.array([1, 2, 3, 4, 5])) + np.random.randn(10)
    # Create an instance of the SGD class
    model = SGD(lr=lr, epochs=epochs, batch_size=batch_size, tol=tol)
    w, b = model.fit(Xx, Yy)
    # Predict using predict method from model
    Y_pred = w * X + b
    return Y_pred


# Adam optimization algorithm
def adam(f, x_ini, bounds, gradf=None, n_iter=30, alpha=.02, beta1=.8, beta2=.99, eps=1e-4):
    if x_ini is None:
        # Generate an initial point
        x = bounds[:, 0] + np.random.rand(len(bounds))\
        * (bounds[:, 1] - bounds[:, 0])
        score = f(x[0], x[1])

    # Initialize Adam moments
    def update_parameters_with_adam(x, grads, s, v, t, lrate=0.01,beta1=0.9, beta2=0.999,eps=1e-8):
        s = beta1 * s + (1.0 - beta1) * grads
        v = beta2 * v + (1.0 - beta2) * grads ** 2
        s_hat = s / (1.0 - beta1 ** (t + 1))
        v_hat = v / (1.0 - beta2 ** (t + 1))
        x = x - lrate * s_hat / (np.sqrt(v_hat) + eps)
        return x, s, v

    # Initialize Adam moments
    s, v = np.zeros(2), np.zeros(2)

    # Run the gradient descent updates
    for t in range(n_iter):
        # Calculate gradient g(t)
        g = gradf(x[0], x[1])

        # Update parameters using Adam
        x, s, v = update_parameters_with_adam(x, g, s,
                                            v, t, alpha,
                                            beta1, beta2,
                                            eps)

        # Evaluate candidate point
        score = f(x[0], x[1])

        # Report progress
        print('>%d = %.5f' % (t, score))

    return [x, score]

# Set the random seed
#np.random.seed(1)
# Define the range for input
#bounds = np.array([[-1.0, 1.0], [-1.0, 1.0]])
#x_ini = bounds[:, 0] + np.random.rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
# Perform the gradient descent search with Adam
#best, score = adam(obj, x_ini, bounds)
