import matplotlib
matplotlib.use('TkAgg')
from matplotlib.lines import Line2D
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
import matplotlib.pylab as plt
from matplotlib import ticker
import matplotlib.pyplot as pylt
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from math import *
from morphomatics.geom import BezierSpline
from morphomatics.manifold import Sphere, CubicBezierfold, Manifold, Euclidean
from morphomatics.stats import ExponentialBarycenter
from PIL import Image
# import pillow
try:
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.basemap import Basemap
except: print('mpl_toolkits not available')
from matplotlib import cm as cm
from scipy.interpolate import CubicSpline
import matplotlib.patches as mpatches
colors = {0: "b", 1: "orange", 2: "r"}
group_0 = mpatches.Patch(color=colors[0], label="Cat 0")
group_1 = mpatches.Patch(color=colors[1], label="Cat 1-3")
group_2 = mpatches.Patch(color=colors[2], label="Cat 4-5")
legend_handle = [group_0, group_1, group_2]
cmap_cat = cm.get_cmap('jet')
#cmap_cat = matplotlib.colormaps['jet']
cnorm_cat = cm.colors.Normalize(vmin=20, vmax=137)
lbl_type = {'TD': 0, 'TS': 1, 'HU': 2, 'EX': 3, 'SD': 4, 'SS': 5, 'LO': 6, 'WV': 7, 'DB': 8}
subj_dict = {'Cyclone Nr': 0, 'Name': 1, 'n_sample': 2}
seq_dict = {'Date': 0, 'Time': 1, 'StatusType': 2, 'coord_2D': [3, 4], 'MaxWind': 5, 'MinPressure': 6, 'coord_3D': [7, 8, 9]}
N_SUBJ, N_SAMPLES = 218, 32  #N_SAMPLES:average = 32, Max = 96
STR_MAXWIND = 'Maximum sustained wind in knots'
# %matplotlib inline
# import imageio
# matplotlib.use("Agg")  # NOQA
eps = 1e-8

def initial_mean(pu, M: Manifold):
    """
    Initialize mean geodesic
    """
    # compute mean of base points
    mean_p = ExponentialBarycenter.compute(M, pu[:, 0])
    # compute mean of tangent vectors
    PT = lambda p, u: M.metric.transp(p, mean_p.estimate_, mean_p)
    mean_v = np.mean([PT(*pu_i) for pu_i in pu], 0)
    return np.array([mean_p, mean_v])


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

def load_data_hur():
    # path = 'hur.csv'
    return pd.read_csv('../datasets/hur.csv', header=None)

def load_splines():
    S2 = Sphere()
    cBfS2 = CubicBezierfold(S2, 1)
    CP_file = np.load('../datasets/splines.npz')

    splines_S2 = []
    max_wind_spline_coefficients = []
    for P, c in zip(CP_file['cubic_CP_S2'], CP_file['coeff_wind']):
        splines_S2.append(cBfS2.from_velocity_representation(P))
        max_wind_spline_coefficients.append(c)

    return splines_S2, max_wind_spline_coefficients

# Earth Science
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
    lat = np.sign(z)*180*np.arctan(z/sqrt(x**2 + y**2))/np.pi
    lon = 180*np.arctan2(y, x)/np.pi # West is negative
    return lat, lon


def visEarth(seq_list, cat_list, title=None, c_map=cmap_cat):
    fig = plt.figure(figsize=(12, 12))
    plt.rcParams['font.size'] = 14
    # set perspective angle
    #lat_viewing_angle,  lon_viewing_angle = 10, -45

    # define color maps for water and land
    ocean_map = (plt.get_cmap('ocean'))(210)
    cmap = plt.get_cmap('gist_earth')

    # call the basemap and use orthographic projection at viewing angle
    m = Basemap(projection='ortho', lat_0=10, lon_0=-45)
    #m = Basemap(projection='lcc', lon_0=-60, lat_0=20, lat_1=45, lat_2=55, width=1.2E7, height=1.0E7)  # conic

    # coastlines, map boundary, fill continents/water, fill ocean, draw countries
    m.drawcoastlines()
    m.drawmapboundary(fill_color=ocean_map)
    m.fillcontinents(color=cmap(200), lake_color=ocean_map)
    m.drawcountries()

    # latitude/longitude line vectors
    lat_line_range, lat_lines = [-90, 90], 8
    lat_line_count = (lat_line_range[1] - lat_line_range[0]) / lat_lines
    merid_range, merid_lines = [-180, 180], 8
    merid_count = (merid_range[1] - merid_range[0]) / merid_lines
    m.drawparallels(np.arange(lat_line_range[0], lat_line_range[1], lat_line_count))
    m.drawmeridians(np.arange(merid_range[0], merid_range[1], merid_count))

    # add points
    # x0, y0 = m(2, 41)
    # m.plot(x0, y0, marker='D', color='r')
    # plt.annotate('Barcelona', xy=(x0, y0), xycoords='data', xytext=(-90, 10), textcoords='offset points',
    #             color='r', arrowprops=dict(arrowstyle="->", color='g'))
    #x0, y0 = m(0, 0)  # origin (lat = lon = 0)
    #m.plot(x0, y0, marker='.', color='b')
    #plt.annotate('origin', xy=(x0, y0))
    if title is not None:
        plt.title(title)
    latlons = []
    if seq_list[0].shape[-1] == 3:
        for seq in seq_list:
            latlon = np.zeros((seq.shape[0], 2))
            for j in range(seq.shape[0]):
                latlon[j, 0], latlon[j, 1] = coord_3D2D(seq[j])
            latlons.append(latlon)
    else:
        latlons = seq_list
    for i in range(len(seq_list)):
        lats, lons, cat = latlons[i][:, 0], latlons[i][:, 1], cat_list[i]
        #lats, lons = seq[:, 0], seq[:, 1]  # [40, 30, 10, 0, 0, -5]  # [-10, -20, -25, -10, 0, 10]
        color = cat #max(cat)*cat/137  # np.array([0, 10, 20, 30, 40, 45, 50, 55, 60, 70, 80, 90, 100])
        norm = cnorm_cat  # gleich: plt.Normalize(vmin=0, vmax=137)
        x, y = m(lons, lats)
        # m.plot(x, y, marker=None, color='m')
        marker = 'o' if i == 0 else '*'
        linewidth = .1 if i == 0 else .5
        s = 16 if i == 0 else 10
        sc = m.scatter(x, y, marker=marker, linewidth=linewidth, s=s, c=color, cmap=c_map, norm=norm)
        #divider = make_axes_locatable(ax)
        #cax = divider.append_axes("right", size="5%", pad=0.1)
        #plt.colorbar(sc, fraction=.05, shrink=.5, label='Maximum sustained wind (in knots)')
        #mappable = cm.ScalarMappable(cnorm,cmap)
    cbar = m.colorbar(mappable=None, location='right', size='5%')
    cbar.set_label('Maximum sustained wind in knots', size=12)
    #plt.colorbar(sc, ax=None, fraction=.05, shrink=.5, label='Maximum sustained wind (in knots)')  # TODO
    plt.clim(0, 137)
    #plt.savefig('orthographic_map_example_python.png', dpi=150, transparent=True)
    #plt.savefig('figures/tracks.png')
    plt.show(block=True)


def sample_spline(B: BezierSpline, n: int = 50) -> np.array:
    """Sample a Bezier spline at n evenly spaced points"""
    return np.array(jax.vmap(B.eval)(jnp.linspace(0, B.nsegments, n)))


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

def plot_3d(points, points_color, title, legend_handle, size=10):
    x, y, z = points.T
    fig, ax = plt.subplots(
        figsize=(size, size),
        facecolor="white",
        tight_layout=True,
        subplot_kw={"projection": "3d"},
    )
    fig.suptitle(title, size=16)
    col = ax.scatter(x, y, z, c=points_color, s=20, alpha=0.8)
    #ax.view_init(azim=-60, elev=9)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.zaxis.set_major_locator(ticker.MultipleLocator(1))
    #fig.colorbar(col, ax=ax, orientation="horizontal", shrink=0.6, aspect=60, pad=0.01)
    #ax.legend(fontsize=12)
    ax.legend(handles=legend_handle)
    plt.show()

def plot_2d(points, points_color, title):
    fig, ax = plt.subplots(figsize=(3, 3), facecolor="white", constrained_layout=True)
    fig.suptitle(title, size=16)
    add_2d_scatter(ax, points, points_color)
    plt.show()


def add_2d_scatter(ax, points, points_color, title=None):
    x, y = points.T
    ax.scatter(x, y, c=points_color, s=50, alpha=0.8)
    ax.set_title(title)
    ax.xaxis.set_major_formatter(ticker.NullFormatter())
    ax.yaxis.set_major_formatter(ticker.NullFormatter())


def generate_on_vec(M, p, u, key):
    """
    Generates a unit-length tangent vector V at point S that is
    orthogonal to vector U under the Riemannian metric of manifold M.

    Parameters:
    -----------
    M : Manifold object (e.g., from geomstats or pymanopt)
    S : array-like
        The point on the manifold where the tangent space resides.
    U : array-like
        The reference tangent vector (e.g., current velocity).

    Returns:
    --------
    V : array-like
        A unit-length tangent vector orthogonal to U.
    """
    z = M.randvec(p, key)
    inner_uz = M.metric.inner(p, u, z)
    inner_uu = M.metric.inner(p, u, u)
    v_raw = z - (inner_uz / (inner_uu + eps)) * u
    # Normalization
    # Scale V_raw so that its Riemannian norm is exactly 1.0
    norm_v = jnp.sqrt(M.metric.inner(p, v_raw, v_raw))
    v = v_raw / (norm_v + 1e-10)

    return v