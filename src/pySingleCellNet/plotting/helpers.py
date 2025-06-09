import numpy as np
from matplotlib.colors import ListedColormap, to_rgb
import matplotlib.tri as mtri
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from matplotlib.colors import to_hex
import matplotlib.pyplot as plt

def _smooth_contour(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    levels: int = 6,
    grid_res: int = 200,
    smooth_sigma: float = 2,
    contour_kwargs: dict = None
):
    """Overlay smooth contour lines by gridding + Gaussian blur.

    Args:
        x, y: 1D arrays of spatial coordinates (length n_obs).
        z:    1D array of normalized or summarized expression (length n_obs).
        levels: Number of contour levels or list of levels.
        grid_res: Resolution of the regular grid along each axis.
        smooth_sigma: Sigma for Gaussian filter to smooth the gridded field.
        contour_kwargs: Extra kwargs passed to plt.contour (e.g. colors, linewidths).

    Returns:
        The contour set drawn on the current axes.
    """
    # 1) create regular grid
    xi = np.linspace(x.min(), x.max(), grid_res)
    yi = np.linspace(y.min(), y.max(), grid_res)
    Xi, Yi = np.meshgrid(xi, yi)

    # 2) interpolate scattered z onto the grid
    Zi = griddata((x, y), z, (Xi, Yi), method='cubic', fill_value=np.nan)

    # 3) smooth the gridded values
    Zi_s = gaussian_filter(Zi, sigma=smooth_sigma, mode='nearest')

    # 4) draw contours
    ctr_kw = {} if contour_kwargs is None else contour_kwargs
    cs = plt.contour(Xi, Yi, Zi_s, levels=levels, **ctr_kw)
    plt.clabel(cs, inline=True, fontsize=8)
    return cs


def make_bivariate_cmap(
    c00: str = "#f0f0f0",
    c10: str = "#e31a1c",
    c01: str = "#1f78b4",
    c11: str = "#ffff00",
    n: int = 128
) -> ListedColormap:
    """Create a bivariate colormap by bilinear‐interpolating four corner colors.

    This builds an (n × n) grid of RGB colors, blending smoothly between
    the specified corner colors:
      - c00 at (low, low)
      - c10 at (high, low)
      - c01 at (low, high)
      - c11 at (high, high)
    
    Args:
        c00: Matplotlib color spec (hex, name, or RGB tuple) for the low/low corner.
        c10: Color for the high/low corner.
        c01: Color for the low/high corner.
        c11: Color for the high/high corner.
        n:   Resolution per axis. The total length of the returned colormap is n*n.

    Returns:
        ListedColormap: A colormap with n*n entries blending between the four corners.
    """
    # Convert corner colors to RGB arrays
    corners = {
        (0, 0): np.array(to_rgb(c00)),
        (1, 0): np.array(to_rgb(c10)),
        (0, 1): np.array(to_rgb(c01)),
        (1, 1): np.array(to_rgb(c11)),
    }
    
    # Build an (n, n, 3) grid by bilinear interpolation
    lut = np.zeros((n, n, 3), dtype=float)
    xs = np.linspace(0, 1, n)
    ys = np.linspace(0, 1, n)
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            lut[j, i] = (
                corners[(0, 0)] * (1 - x) * (1 - y) +
                corners[(1, 0)] * x       * (1 - y) +
                corners[(0, 1)] * (1 - x) * y       +
                corners[(1, 1)] * x       * y
            )
    
    # Flatten to (n*n, 3) and return as a ListedColormap
    return ListedColormap(lut.reshape(n * n, 3))
