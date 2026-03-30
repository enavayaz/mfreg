from util import visSphere
import numpy as np
from morphomatics.manifold import Sphere

data = np.load('datasets/sphElse.npz', allow_pickle=True)
yt1, yp1 = data['ytest'], data['ypred']

data = np.load('datasets/sphPoly.npz', allow_pickle=True)
yt2, yp2 = data['ytest'], data['ypred']
N = np.cross(yt2[0], yt2[-1])
N = N/np.linalg.norm(N)
R = np.eye(3) - 2*N.reshape(3, 1)@N.reshape(1,3)
for k in range(len(yp2)-1):
    yt2[k] = yt2[k]@R
    yp2[k] = yp2[k]@R
visSphere([yt1, yt2[:-1], yp1[:-1], yp2], ['b', 'b', 'r', 'r'])

P = np.array([
    [-4, 0, -1],
    [2, 1, 1],
    [-2, 0, -1],
    [0, 1, -1]
])
N = np.cross(P[0], P[-1])
N = N/np.linalg.norm(N)
R = np.eye(3) - 2*N.reshape(3, 1)@N.reshape(1,3)
for k in range(len(P)):
    P[k] = P[k]@R

M = Sphere()
def bez_sph(n_points):
# 1. Define Control Points and Parameters
    P = np.array([
        [-2, 0, -1],
        [0, 1, 0],
        [-1, 0, -1],
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

y= bez_sph(50)
visSphere([y,y],['b','r'])