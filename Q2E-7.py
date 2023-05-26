# Matthew Lisondra
# ME8135 Q2E

# Import the necessary modules
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
from scipy.stats import multivariate_normal
import math

from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

# sourced from https://matplotlib.org/stable/gallery/statistics/confidence_ellipse.html
def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

# Set the p and theta and calculate y mean
p=1
theta=0.00872665 # 5 deg
y_mean = (p*np.cos(theta),p*np.sin(theta))

# Find the y covariance
x_cov = [[0.01,0],[0,0.1]]
J = [[np.cos(theta), np.sin(theta)],[0-p*np.sin(theta),p*np.cos(theta)]]
y_cov = np.array(J) * np.array(x_cov) * np.transpose(np.array(J))

# Sample 1000 points from the y distribution
y = np.random.multivariate_normal(y_mean, y_cov, size=1000)

# Plot the y distribution with uncertainty ellipse
fig, ax_kwargs = plt.subplots(figsize=(6, 6))
plt.plot(y[:, 0], y[:, 1], '.', alpha=0.5, label="y=f(x)")
ax_kwargs.axvline(c='grey', lw=1)
ax_kwargs.axhline(c='grey', lw=1)
confidence_ellipse(y[:, 0], y[:, 1], ax_kwargs, edgecolor='red')
plt.xlabel('x') 
plt.ylabel('y')
plt.title("Monte Carlo Simulation on Polar-Coord -> Cartesian")
plt.legend()
plt.axis('equal')
plt.grid()
plt.show()

