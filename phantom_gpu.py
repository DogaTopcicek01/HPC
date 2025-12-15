import cupy as cp

def gen_shepp_logan(num_rows, num_cols):
    """
    Generate a Shepp Logan phantom (GPU / CuPy).
    """

    sl_paras = [
        {'x0': 0.0, 'y0': 0.0, 'a': 0.69, 'b': 0.92, 'theta': 0, 'gray_level': 2.0},
        {'x0': 0.0, 'y0': -0.0184, 'a': 0.6624, 'b': 0.874, 'theta': 0, 'gray_level': -0.98},
        {'x0': 0.22, 'y0': 0.0, 'a': 0.11, 'b': 0.31, 'theta': -18, 'gray_level': -0.02},
        {'x0': -0.22, 'y0': 0.0, 'a': 0.16, 'b': 0.41, 'theta': 18, 'gray_level': -0.02},
        {'x0': 0.0, 'y0': 0.35, 'a': 0.21, 'b': 0.25, 'theta': 0, 'gray_level': 0.01},
        {'x0': 0.0, 'y0': 0.1, 'a': 0.046, 'b': 0.046, 'theta': 0, 'gray_level': 0.01},
        {'x0': 0.0, 'y0': -0.1, 'a': 0.046, 'b': 0.046, 'theta': 0, 'gray_level': 0.01},
        {'x0': -0.08, 'y0': -0.605, 'a': 0.046, 'b': 0.023, 'theta': 0, 'gray_level': 0.01},
        {'x0': 0.0, 'y0': -0.605, 'a': 0.023, 'b': 0.023, 'theta': 0, 'gray_level': 0.01},
        {'x0': 0.06, 'y0': -0.605, 'a': 0.023, 'b': 0.046, 'theta': 0, 'gray_level': 0.01},
    ]

    axis_x = cp.linspace(-1.0, 1.0, num_cols, dtype=cp.float32)
    axis_y = cp.linspace(1.0, -1.0, num_rows, dtype=cp.float32)

    x_grid, y_grid = cp.meshgrid(axis_x, axis_y, indexing="xy")
    image = cp.zeros_like(x_grid, dtype=cp.float32)

    for el in sl_paras:
        theta = el["theta"] * cp.pi / 180.0
        image += _gen_ellipse(
            x_grid=x_grid,
            y_grid=y_grid,
            x0=el["x0"],
            y0=el["y0"],
            a=el["a"],
            b=el["b"],
            theta=theta,
            gray_level=el["gray_level"],
        )

    return image


def gen_microscopy_sample(num_rows, num_cols):
    """
    Generate a microscopy sample phantom.

    Args:
        num_rows: int, number of rows.
        num_cols: int, number of cols.

    Return:
        out_image: 2D array, num_rows*num_cols
    """

    # The function describing the phantom is defined as the sum of 8 ellipses inside a 2×4 rectangle:
    ms_paras = [
        {'x0': 0.0, 'y0': -0.0184, 'a': 0.6624, 'b': 1.748, 'theta': 0, 'gray_level': 0.2},
        {'x0': -0.1, 'y0': 1.343, 'a': 0.11, 'b': 0.10, 'theta': 0, 'gray_level': 0.8},
        {'x0': 0.0, 'y0': 0.9, 'a': 0.33, 'b': 0.15, 'theta': 0, 'gray_level': 0.4},
        {'x0': 0.25, 'y0': 0.4, 'a': 0.1, 'b': 0.2, 'theta': 0, 'gray_level': 0.8},
        {'x0': -0.2, 'y0': 0.0, 'a': 0.2, 'b': 0.08, 'theta': 0, 'gray_level': 0.4},
        {'x0': 0.2, 'y0': -0.35, 'a': 0.1, 'b': 0.1, 'theta': 0, 'gray_level': 0.8},
        {'x0': 0.25, 'y0': -0.8, 'a': 0.2, 'b': 0.08, 'theta': 0, 'gray_level': 0.8},
        {'x0': -0.04, 'y0': -1.3, 'a': 0.33, 'b': 0.15, 'theta': 0, 'gray_level': 0.8}
    ]
    axis_x = cp.linspace(-1, 1, num_cols)
    axis_y = cp.linspace(2, -2, num_rows)

    x_grid, y_grid = cp.meshgrid(axis_x, axis_y )
    image = x_grid * 0.0

    for el_paras in ms_paras:
        image += _gen_ellipse(x_grid=x_grid, y_grid=y_grid, x0=el_paras['x0'], y0=el_paras['y0'],
                             a=el_paras['a'], b=el_paras['b'], theta=el_paras['theta'] / 180.0 * cp.pi,
                             gray_level=el_paras['gray_level'])

    return image


def gen_shepp_logan_3d(num_rows, num_cols, num_slices, block_size=(2,2,2), scale=1.0, offset_x=0.0, offset_y=0.0, offset_z=0.0):
    """
    Generate a 3D Shepp Logan phantom.
    Optional arguments can be used to control the scale and position,
    and block average smoothing is used to reduce aliasing artifacts.

    Args:
        num_rows: int, number of rows in generated phantom.
        num_cols: int, number of columns in generated phantom.
        num_slices: int, number of slices in generated phantom.

        block_size: (int, int, int): size of block average.
        scale: (scalar, optional), scaling factor of phantom within the image. from 0 to 1
        offset_x: scalar, proportion of x-axis that is added to x-coordinate of phantom within image
        offset_y: scalar, proportion of y-axis that is added to y-coordinate of phantom within image
        offset_z: scalar, proportion of z-axis that is added to z-coordinate of phantom within image

    Return:
        out_image: 3D array, num_slices*num_rows*num_cols
    """

    phantom_raw = gen_shepp_logan_3d_raw(num_rows*block_size[1], num_cols*block_size[2], num_slices*block_size[0],
                                         scale=scale, offset_x=offset_x, offset_y=offset_y, offset_z=offset_z)
    phantom = phantom_raw.reshape(phantom_raw.shape[0]//block_size[0], block_size[0],
                                  phantom_raw.shape[1]//block_size[1], block_size[1],
                                  phantom_raw.shape[2]//block_size[2], block_size[2]).sum((1, 3, 5)) / (block_size[0]*block_size[1]*block_size[2])
    return phantom


def gen_shepp_logan_3d_raw(num_rows, num_cols, num_slices, scale=1.0, offset_x=0.0, offset_y=0.0, offset_z=0.0):
    """
    Generate a 3D Shepp Logan phantom based on below reference.

    Kak AC, Slaney M. Principles of computerized tomographic imaging. Page.102. IEEE Press, New York, 1988. https://engineering.purdue.edu/~malcolm/pct/CTI_Ch03.pdf

    Args:
        num_rows: int, number of rows.
        num_cols: int, number of cols.
        num_slices: int, number of slices.

        scale: (scalar, optional), scaling factor of phantom within the image. from 0 to 1
        offset_x: scalar, proportion of x-axis that is added to x-coordinate of phantom within image
        offset_y: scalar, proportion of y-axis that is added to y-coordinate of phantom within image
        offset_z: scalar, proportion of z-axis that is added to z-coordinate of phantom within image

    Return:
        out_image: 3D array, num_slices*num_rows*num_cols
    """

    # The function describing the phantom is defined as the sum of 10 ellipsoids inside a 2×2×2 cube:
    sl3d_paras = [
        {'x0': 0.0, 'y0': 0.0, 'z0': 0.0, 'a': 0.69, 'b': 0.92, 'c': 0.9, 'gamma': 0, 'gray_level': 2.0},
        {'x0': 0.0, 'y0': 0.0, 'z0': 0.0, 'a': 0.6624, 'b': 0.874, 'c': 0.88, 'gamma': 0, 'gray_level': -0.98},
        {'x0': -0.22, 'y0': 0.0, 'z0': -0.25, 'a': 0.41, 'b': 0.16, 'c': 0.21, 'gamma': 108, 'gray_level': -0.02},
        {'x0': 0.22, 'y0': 0.0, 'z0': -0.25, 'a': 0.31, 'b': 0.11, 'c': 0.22, 'gamma': 72, 'gray_level': -0.02},
        {'x0': 0.0, 'y0': 0.35, 'z0': -0.25, 'a': 0.21, 'b': 0.25, 'c': 0.5, 'gamma': 0, 'gray_level': 0.02},
        {'x0': 0.0, 'y0': 0.1, 'z0': -0.25, 'a': 0.046, 'b': 0.046, 'c': 0.046, 'gamma': 0, 'gray_level': 0.02},
        {'x0': -0.08, 'y0': -0.65, 'z0': -0.25, 'a': 0.046, 'b': 0.023, 'c': 0.02, 'gamma': 0, 'gray_level': 0.01},
        {'x0': 0.06, 'y0': -0.65, 'z0': -0.25, 'a': 0.046, 'b': 0.023, 'c': 0.02, 'gamma': 90, 'gray_level': 0.01},
        {'x0': 0.06, 'y0': -0.105, 'z0': 0.625, 'a': 0.056, 'b': 0.04, 'c': 0.1, 'gamma': 90, 'gray_level': 0.02},
        {'x0': 0.0, 'y0': 0.1, 'z0': 0.625, 'a': 0.056, 'b': 0.056, 'c': 0.1, 'gamma': 0, 'gray_level': -0.02}
    ]

    axis_x = cp.linspace(-1.0, 1.0, num_cols)
    axis_y = cp.linspace(1.0, -1.0, num_rows)
    axis_z = cp.linspace(-1.0, 1.0, num_slices)

    x_grid, y_grid, z_grid = cp.meshgrid(axis_x, axis_y, axis_z)
    image = x_grid * 0.0

    shift_x = offset_x * 2.0
    shift_y = offset_y * 2.0
    shift_z = offset_z * 2.0

    for el_paras in sl3d_paras:
        image += _gen_ellipsoid(x_grid=x_grid, y_grid=y_grid, z_grid=z_grid, x0=el_paras['x0']*scale - shift_x, y0=el_paras['y0']*scale - shift_y,
                               z0=el_paras['z0']*scale - shift_z,
                               a=el_paras['a']*scale, b=el_paras['b']*scale, c=el_paras['c']*scale,
                               gamma=el_paras['gamma'] / 180.0 * cp.pi,
                               gray_level=el_paras['gray_level'])



    return cp.transpose(image, (2, 0, 1))


def gen_microscopy_sample_3d(num_rows, num_cols, num_slices):
    """
    Generate a 3D microscopy sample phantom.

    Args:
        num_rows: int, number of rows.
        num_cols: int, number of cols.
        num_slices: int, number of slices.

    Return:
        out_image: 3D array, num_slices*num_rows*num_cols
    """

    # The function describing the phantom is defined as the sum of 8 ellipsoids inside a 2×4×2 cuboid:
    ms3d_paras = [
        {'x0': 0.0, 'y0': -0.0184, 'z0':0.0, 'a': 0.6624, 'b': 1.748, 'c':0.8, 'gamma': 0, 'gray_level': 0.2},
        {'x0': -0.1, 'y0': 1.343, 'z0':0.0, 'a': 0.11, 'b': 0.10, 'c':0.20, 'gamma': 0, 'gray_level': 0.8},
        {'x0': 0.0, 'y0': 0.9, 'z0':0.0, 'a': 0.33, 'b': 0.15, 'c':0.66, 'gamma': 0, 'gray_level': 0.4},
        {'x0': 0.25, 'y0': 0.4, 'z0':0.0, 'a': 0.1, 'b': 0.2, 'c':0.40, 'gamma': 0, 'gray_level': 0.8},
        {'x0': -0.2, 'y0': 0.0, 'z0':0.0, 'a': 0.2, 'b': 0.08, 'c':0.40, 'gamma': 0, 'gray_level': 0.4},
        {'x0': 0.2, 'y0': -0.35, 'z0':0.0, 'a': 0.1, 'b': 0.1, 'c':0.2, 'gamma': 0, 'gray_level': 0.8},
        {'x0': 0.25, 'y0': -0.8, 'z0':0.0, 'a': 0.2, 'b': 0.08, 'c':0.4, 'gamma': 0, 'gray_level': 0.8},
        {'x0': -0.04, 'y0': -1.3, 'z0':0.0, 'a': 0.33, 'b': 0.15, 'c':0.30, 'gamma': 0, 'gray_level': 0.8}
    ]

    axis_x = cp.linspace(-1.0, 1.0, num_cols)
    axis_y = cp.linspace(2.0, -2.0, num_rows)
    axis_z = cp.linspace(-1.0, 1.0, num_slices)

    x_grid, y_grid, z_grid = cp.meshgrid(axis_x, axis_y, axis_z)
    image = x_grid * 0.0

    for el_paras in ms3d_paras:
        image += _gen_ellipsoid(x_grid=x_grid, y_grid=y_grid, z_grid=z_grid, x0=el_paras['x0'], y0=el_paras['y0'],
                               z0=el_paras['z0'],
                               a=el_paras['a'], b=el_paras['b'], c=el_paras['c'],
                               gamma=el_paras['gamma'] / 180.0 * cp.pi,
                               gray_level=el_paras['gray_level'])

    return cp.transpose(image, (2, 0, 1))


def gen_lamino_sample_3d(num_rows, num_cols, num_slices, tile_rows=1, tile_cols=1):
    """
    Generate a 3D laminography phantom with a tiling of custom size.
    Args:
        num_rows: int, number of rows.
        num_cols: int, number of cols.
        num_slices: int, number of slices.
        tile_rows: int, [Default=1] number of rows in the phantom tiling
        tile_cols: int, [Default=1] number of cols in the phantom tiling
    Return:
        out_image: 3D array, num_slices x num_rows * (1+tile_rows) x num_cols * (1+tile_cols)
    """

    # The function describing the phantom is defined as the sum of 9 ellipsoids inside a 4x4x1 cuboid:
    ms3d_paras = [
        {'x0': -0.2, 'y0': 1.343, 'z0': 0.0, 'a': 0.22, 'b': 0.10, 'c': 0.10, 'gamma': 0, 'gray_level': 0.8},
        {'x0': 0.0, 'y0': 0.9, 'z0': 0.0, 'a': 0.66, 'b': 0.15, 'c': 0.33, 'gamma': 0, 'gray_level': 0.4},
        {'x0': 0.5, 'y0': 0.4, 'z0': 0.0, 'a': 0.2, 'b': 0.2, 'c': 0.20, 'gamma': 0, 'gray_level': 0.8},
        {'x0': -0.4, 'y0': 0.0, 'z0': 0.0, 'a': 0.4, 'b': 0.08, 'c': 0.20, 'gamma': 0, 'gray_level': 0.4},
        {'x0': 0.4, 'y0': -0.35, 'z0': 0.0, 'a': 0.2, 'b': 0.1, 'c': 0.1, 'gamma': 0, 'gray_level': 0.8},
        {'x0': 0.5, 'y0': -0.8, 'z0': 0.0, 'a': 0.4, 'b': 0.08, 'c': 0.2, 'gamma': 0, 'gray_level': 0.8},
        {'x0': -0.08, 'y0': -1.3, 'z0': 0.0, 'a': 0.66, 'b': 0.15, 'c': 0.15, 'gamma': 0, 'gray_level': 0.8},
        {'x0': -1.5, 'y0': 0.3, 'z0': 0.0, 'a': 0.2, 'b': 0.7, 'c': 0.20, 'gamma': 0, 'gray_level': 0.8},
        {'x0': 1.5, 'y0': -0.5, 'z0': 0.0, 'a': 0.3, 'b': 0.5, 'c': 0.30, 'gamma': 0, 'gray_level': 0.4}
    ]

    axis_x = cp.linspace(-2.0, 2.0, num_cols)
    axis_y = cp.linspace(2.0, -2.0, num_rows)
    axis_z = cp.linspace(-0.5, 0.5, num_slices)

    x_grid, y_grid, z_grid = cp.meshgrid(axis_x, axis_y, axis_z)
    image = x_grid * 0.0
    image += 0.1

    for el_paras in ms3d_paras:
        image += _gen_ellipsoid(x_grid=x_grid, y_grid=y_grid, z_grid=z_grid, x0=el_paras['x0'], y0=el_paras['y0'],
                                z0=el_paras['z0'],
                                a=el_paras['a'], b=el_paras['b'], c=el_paras['c'],
                                gamma=el_paras['gamma'] / 180.0 * cp.pi,
                                gray_level=el_paras['gray_level'])

    image = cp.transpose(image, (2, 0, 1))

    # Use tiling to create a larger phantom
    image = cp.tile(image, (1, tile_rows, tile_cols))

    return image


def nrmse(image, reference_image):
    """
    Compute the normalized root mean square error between image and reference_image.

    Args:
        image: Calculated image
        reference_image: Ground truth image

    Returns:
        Root mean square of (image - reference_image) divided by RMS of reference_image

    """
    rmse = cp.sqrt(((image - reference_image) ** 2).mean())
    denominator = cp.sqrt(((reference_image) ** 2).mean())

    return rmse/denominator


def _gen_ellipse(x_grid, y_grid, x0, y0, a, b, gray_level, theta=0):
    """
    Returns an image with a 2D ellipse in a 2D plane with a center of [x0,y0] and ...

    Args:
        x_grid(float): 2D grid of X coordinate values
        y_grid(float): 2D grid of Y coordinate values
        x0(float): horizontal center of ellipse.
        y0(float): vertical center of ellipse.
        a(float): X-axis radius.
        b(float): Y-axis radius.
        gray_level(float): Gray level for the ellipse.
        theta(float): [Default=0.0] counter-clockwise angle of rotation in radians

    Return:
        ndarray: 2D array with the same shape as x_grid and y_grid.

    """
    image = (((x_grid - x0) * cp.cos(theta) + (y_grid - y0) * cp.sin(theta)) ** 2 / a ** 2
             + ((x_grid - x0) * cp.sin(theta) - (y_grid - y0) * cp.cos(theta)) ** 2 / b ** 2 <= 1.0) * gray_level

    return image


def _gen_ellipsoid(x_grid, y_grid, z_grid,
                   x0, y0, z0, a, b, c,
                   gray_level, alpha=0, beta=0, gamma=0):

    # Explicit CuPy scalars (IMPORTANT)
    one = cp.array(1.0, dtype=cp.float32)
    zero = cp.array(0.0, dtype=cp.float32)

    rx = cp.array([
        [one,  zero, zero],
        [zero, cp.cos(-alpha), -cp.sin(-alpha)],
        [zero, cp.sin(-alpha),  cp.cos(-alpha)]
    ])

    ry = cp.array([
        [cp.cos(-beta), zero, cp.sin(-beta)],
        [zero, one, zero],
        [-cp.sin(-beta), zero, cp.cos(-beta)]
    ])

    rz = cp.array([
        [cp.cos(-gamma), -cp.sin(-gamma), zero],
        [cp.sin(-gamma),  cp.cos(-gamma), zero],
        [zero, zero, one]
    ])

    r = rx @ ry @ rz

    cor = cp.stack([
        x_grid.ravel() - x0,
        y_grid.ravel() - y0,
        z_grid.ravel() - z0
    ])

    expr = (
        (r[0] @ cor) ** 2 / a ** 2 +
        (r[1] @ cor) ** 2 / b ** 2 +
        (r[2] @ cor) ** 2 / c ** 2
    )

    image = (expr <= 1.0) * gray_level

    return image.reshape(x_grid.shape)
