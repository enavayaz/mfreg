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


def visSphere(points_list, color_list, size=20, nice=True):
    """
    Visualize groups of points on the 2D-sphere
    """
    import matplotlib.pyplot as plt
    _ = plt.figure(figsize=(size, size))
    ax = plt.subplot(111, projection="3d")
    ax.set_box_aspect([1.0, 1.0, 1.0])
    # draw sphere
    if nice:
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = 1 * np.outer(np.cos(u), np.sin(v))
        y = 1 * np.outer(np.sin(u), np.sin(v))
        z = 1 * np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(x, y, z, color='#E8F4F8', alpha=0.25,
                        linewidth=0, antialiased=True, shade=True,
                        lightsource=plt.matplotlib.colors.LightSource(azdeg=45, altdeg=45))
        ax.set_axis_off()
        ax.grid(False)
    else:
        u, v = np.meshgrid(np.linspace(0.0, 2 * np.pi, 40), np.linspace(0.0, np.pi, 20))
        x, y, z = np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v)
        ax.plot_wireframe(x, y, z, color="grey", alpha=0.2)
    for i in range(len(points_list)):
        marker_size = 25 if i == 0 else 15
        for points in points_list[i]:
            points = np.array(points)
            ax.scatter(points[0], points[1], points[2], s=marker_size, color=color_list[i], marker=".")
            if i == 0:
                ax.plot(points[0], points[1], points[2], color='blue', linewidth=0.5, alpha=0.5)
                #ax.scatter(points[0], points[1], points[2], s=25, color=color_list[i], marker=".")
    plt.show(block=True)

from util import visSphere as visx
path = '../datasets/visPoly.npz'
data = np.load(path, allow_pickle=True)
y, yols, yridge = data['ytest'].tolist(), data['yols'].tolist(), data['yriddge'].tolist()

ypoly, ypolyols, ypolyridge = np.array(y), np.array(yols)[0], np.array(yridge)[0]

path = '../datasets/visElse.npz'
data = np.load(path, allow_pickle=True)
y, yols, yridge = data['ytest'].tolist(), data['yols'].tolist(), data['yriddge'].tolist()
yelse, yelseols, yelseridge = np.array(y), np.array(yols)[0], np.array(yridge)[0]

visx([y, yols, yridge], ['b', 'r', 'g'], size=25)