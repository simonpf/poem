"""
poem.geometry
=============

This module provides functions for calculating lines of sight through the atmosphere.
"""
from typing import List, Tuple

import numpy as np
import scipy
import torch
from tqdm import tqdm
import xarray as xr


SEM_A = 6_378_137.0
SEM_B = 6_356_752.0
ECC2 = 1.0 - (SEM_B ** 2 / SEM_A ** 2)


def lla_to_ecef(coords_lla: np.ndarray):
    """
    Converts latitude-longitude-altitude (LLA) coordinates to
    earth-centric earth-fixed coordinates (ECEF)

    Params:
        coords_lla: A numpy.ndarray containing the three coordinates oriented along the last axis.

    Return:
        coords_ecef: An array of the same shape as 'coords_lla' but containing the x, y, and z
             coordinates along the last axis.
    """
    lon = np.radians(coords_lla[..., 0])
    lat = np.radians(coords_lla[..., 1])
    alt = coords_lla[..., 2]

    roc = SEM_A / np.sqrt(1 - ECC2 * np.sin(lat)**2)

    x = (roc + alt) * np.cos(lat) * np.cos(lon)
    y = (roc + alt) * np.cos(lat) * np.sin(lon)
    z = (roc * (1 - ECC2) + alt) * np.sin(lat)

    return np.stack((x, y, z), -1)


def ecef_to_lla(coords_ecef):
    """
    Converts ECEF coordinates back to LLA coordinates.

    Params:
        coords_ecef: A numpy.ndarray containing the coordinates along the last axis.

    Return:
        coords_lla: A numpy.ndarray of the same shape as 'coords_ecef' containing
            the longitude, latitude, and altitude along tis last axis.
    """
    lon = np.arctan2(coords_ecef[..., 1], coords_ecef[..., 0])
    lon = np.nan_to_num(lon, nan=0.0)
    lon = np.degrees(lon)

    p = np.sqrt(coords_ecef[..., 0]**2 + coords_ecef[..., 1]**2)

    lat = np.arctan2(coords_ecef[..., 2], p * (1 - ECC2))
    lat_prev = lat
    roc = SEM_A / np.sqrt(1 - ECC2 * np.sin(lat)**2)
    alt = p / np.cos(lat) - roc
    lat = np.arctan2(coords_ecef[..., -1], p * (1 - ECC2 * (roc / (roc + alt))))


    while np.max(np.abs(lat - lat_prev)) > 1e-6:
        lat_prev = lat
        roc = SEM_A / np.sqrt(1 - ECC2 * np.sin(lat)**2)
        alt = p / np.cos(lat) - roc
        lat = np.arctan2(coords_ecef[..., 2], p * (1 - ECC2 * (roc / (roc + alt))))

    roc = SEM_A / np.sqrt(1 - ECC2 * np.sin(lat)**2)
    alt = p / np.cos(lat) - roc
    lat = np.degrees(lat)

    if not isinstance(lat, np.ndarray):
        if np.isclose(p, 0.0):
            alt = coords_ecef[..., -1]
            lat = np.sign(alt) * 90
            alt = np.abs(alt) - SEM_B
    else:
        mask = np.isclose(p, 0.0)
        alt[mask] = coords_ecef[mask, -1]
        lat[mask] = np.sign(alt[mask]) * 90
        alt[mask] = np.abs(alt[mask]) - SEM_B

    return np.stack([lon, lat, alt], -1)


def calculate_surface_intersection(
        sensor_pos_ecef: np.ndarray,
        sensor_los_ecef: np.ndarray,
):
    coeff_a = (
        sensor_los_ecef[..., 0] ** 2 / SEM_A ** 2 +
        sensor_los_ecef[..., -2] ** 2 / SEM_A ** 2 +
        sensor_los_ecef[..., -1] ** 2 / SEM_B ** 2
    )
    coeff_b = 2.0 * (
        sensor_pos_ecef[..., 0] * sensor_los_ecef[..., 0] / SEM_A ** 2 +
        sensor_pos_ecef[..., 1] * sensor_los_ecef[..., 1] / SEM_A ** 2 +
        sensor_pos_ecef[..., 2] * sensor_los_ecef[..., 2] / SEM_B ** 2
    )
    coeff_c = (
        sensor_pos_ecef[..., 0] ** 2 / SEM_A ** 2 +
        sensor_pos_ecef[..., 1] ** 2 / SEM_A ** 2 +
        sensor_pos_ecef[..., 2] ** 2 / SEM_B ** 2
    ) - 1.0

    discr = coeff_b ** 2 - 4.0 * coeff_a * coeff_c
    root_1 = (np.sqrt(discr) - coeff_b) / (2.0 * coeff_a)
    root_2 = (-np.sqrt(discr) - coeff_b) / (2.0 * coeff_a)

    fac = np.minimum(root_1, root_2)
    pos = sensor_pos_ecef + fac * sensor_los_ecef
    return pos


class EuclideanGrid:
    """
    A uniform Euclidean grid positioned on the surface of the Earth so that_points_native = atm.
    all of its lower corners are above or at the surface.
    """
    def __init__(
            self,
            center_ecef: np.ndarray,
            orientation_ecef: np.ndarray,
            resolution: Tuple[float],
            shape: Tuple[int],
    ):
        z = center_ecef / np.linalg.norm(center_ecef)
        x = orientation_ecef
        x = x - (z * x).sum() * z
        x = x / np.linalg.norm(x)

        y = np.cross(x, z)

        self.x = x
        self.y = y
        self.z = z

        self.center = center_ecef
        self.resolution = np.array(resolution)
        self.shape = np.array(shape)

        ll_ecef = self.center - 0.5 * self.extent[0] * self.x - 0.5 * self.extent[1] * self.y
        lr_ecef = ll_ecef + self.extent[1] * self.y
        rr_ecef = ll_ecef + self.extent[0] * self.x
        rl_ecef = ll_ecef - self.extent[1] * self.y
        ll_lla = ecef_to_lla(ll_ecef)
        lr_lla = ecef_to_lla(lr_ecef)
        rr_lla = ecef_to_lla(rr_ecef)
        rl_lla = ecef_to_lla(rl_ecef)
        min_alt = min([ll_lla[-1], lr_lla[-1], rr_lla[-1], rl_lla[-1]])

        self.center = self.center - min_alt * self.z
        ll_ecef = self.center - 0.5 * self.extent[0] * self.x - 0.5 * self.extent[1] * self.y
        self.xx = ll_ecef[None] + self.x[None] * self.resolution[0] * np.arange(self.shape[0])[..., None]
        self.yy = ll_ecef[None] + self.y[None] * self.resolution[1] * np.arange(self.shape[1])[..., None]
        self.zz = ll_ecef[None] + self.z[None] * self.resolution[2] * np.arange(self.shape[2])[..., None]


    @property
    def extent(self) -> np.ndarray:
        """
        A length-3 vector containing the dimensions of the grid in x, y, and z directions.
        """
        return self.resolution * self.shape


    def ecef_to_native(self, coords_ecef: np.ndarray) -> np.ndarray:
        """
        Convert ECEF coordinates to native coordinates of the grid.

        Args:
            coords_ecef: An array containing ECEF coordinates oriented along the last dimension.

        Return:
            An array of similar shape as 'coords_ecef' containing the coordinates converted
            to native grid coordinates.
        """
        lll_ecef = self.center - 0.5 * self.extent[0] * self.x - 0.5 * self.extent[1] * self.y
        translated = coords_ecef - lll_ecef
        x = (translated * self.x).sum(-1) / self.resolution[0]
        y = (translated * self.y).sum(-1) / self.resolution[1]
        z = (translated * self.z).sum(-1) / self.resolution[2]
        return np.stack((x, y, z), -1)


    def calculate_path_integral_matrices(
            self,
            origin_ecef: np.ndarray,
            los_ecef: np.ndarray,
            step_length: float,
            n_steps: int,
            resolution: int = 100
    ) -> List[torch.Tensor]:
        """
        Calculates matrices that calculate path integrals through the atmospheric field.

        Args:
            origin_ecef: A [n_los, 3] numpy.ndarray defining the origins of the lines of sight.
            los_ecef: A [n_los, 3] numpy.ndarray defining the lines of sight.
            step_length: The length of the steps along the path.
            n_steps: The number of steps.
            resolution: The number of quadrature step to apply to each step along the pencil beam.

        Return:
            A numpy.ndarray of shape [n_los, n_steps, atm_size] containing the matrices to extract
            path integrals from flattened atmospheric fields.
        """

        origin_ecef = origin_ecef.reshape(-1, origin_ecef.shape[-1])
        los_ecef = los_ecef.reshape(-1, los_ecef.shape[-1])
        n_los = origin_ecef.shape[0]

        los_ecef = los_ecef.astype("float64")
        los_ecef = los_ecef / np.linalg.norm(los_ecef, axis=-1, keepdims=True) * step_length / resolution

        origin_native = self.ecef_to_native(origin_ecef)
        step_native = self.ecef_to_native(origin_ecef + los_ecef) - origin_native

        curr_pos = origin_native

        max_inds = np.array([shp - 2 for shp in self.shape])
        flat_size = np.prod(self.shape)

        step_matrices = []

        for step in tqdm(range(n_steps)):

            step_indices = np.zeros((n_los, resolution + 1, 8), np.int64)
            step_weights = np.zeros((n_los, resolution + 1, 8), np.float32)

            for int_step in range(resolution + 1):

                lll = np.maximum(np.minimum(np.trunc(curr_pos).astype(np.int64), max_inds), 0)
                bcoords = np.clip(curr_pos - lll, 0, 1)

                lll_x = lll[..., 0]
                lll_y = lll[..., 1]
                lll_z = lll[..., 2]

                weight_ind  = 0
                for offset_x in range(2):
                    for offset_y in range(2):
                        for offset_z in range(2):
                            inds = np.ravel_multi_index(
                                (lll_x + offset_x, lll_y + offset_y, lll_z + offset_z),
                                self.shape
                            )
                            weights = (
                                (1.0 - bcoords[..., 0] - offset_x + 2.0 * offset_x * bcoords[..., 0]) *
                                (1.0 - bcoords[..., 1] - offset_y + 2.0 * offset_y * bcoords[..., 1]) *
                                (1.0 - bcoords[..., 2] - offset_z + 2.0 * offset_z * bcoords[..., 2])
                            )
                            step_indices[:, int_step, weight_ind] = inds
                            if int_step == 0 or int_step == resolution:
                                step_weights[:, int_step, weight_ind] = 0.5 * weights / resolution
                            else:
                                step_weights[:, int_step, weight_ind] = weights / resolution
                            weight_ind += 1

                curr_pos += step_native

            rows = np.broadcast_to(np.arange(n_los)[:, None, None], step_indices.shape)
            coords = torch.tensor(np.stack((rows.flatten(), step_indices.flatten())))
            step_mat = torch.sparse_coo_tensor(
                coords, torch.tensor(step_weights.flatten()), torch.Size((n_los, flat_size))
            )
            step_mat = step_mat.coalesce()
            step_matrices.append(step_mat)
            curr_pos -= step_native

        return step_matrices


    def calculate_surface_interpolation_matrices(
            self,
            origin_ecef: np.ndarray,
    ) -> np.ndarray:
        """
        Calculates matrices that extract surface values from the surface fields.

        Args:
            origin_ecef: A [n_los, 3] numpy.ndarray defining the origins of the lines of sight.

        Return:
            A numpy.ndarray of shape [n_los, atm_flat] that can be used to calculate
        """
        origin_ecef = origin_ecef.reshape(-1, origin_ecef.shape[-1])
        n_los = origin_ecef.shape[0]
        origin_native = self.ecef_to_native(origin_ecef)

        max_inds = np.array([shp - 2 for shp in self.shape])
        flat_size = np.prod(self.shape[:2])

        lll = np.maximum(np.minimum(np.trunc(origin_native).astype(np.int64), max_inds), 0)
        bcoords = np.clip(origin_native - lll, 0, 1)
        weights = np.zeros((n_los, 4), dtype="float32")
        indices = np.zeros((n_los, 4), dtype="int64")

        lll_x = lll[..., 0]
        lll_y = lll[..., 1]
        lll_z = lll[..., 2]

        weight_ind = 0
        for offset_x in range(2):
            for offset_y in range(2):
                inds = np.ravel_multi_index(
                    (lll_x + offset_x, lll_y + offset_y),
                    self.shape[:2]
                )
                indices[:, weight_ind] = inds
                interp_weights = (
                    (1.0 - bcoords[..., 0] - offset_x + 2.0 * offset_x * bcoords[..., 0]) *
                    (1.0 - bcoords[..., 1] - offset_y + 2.0 * offset_y * bcoords[..., 1])
                )
                weights[:, weight_ind] = interp_weights
                weight_ind += 1

        rows = np.broadcast_to(np.arange(n_los)[:, None], indices.shape)
        coords = torch.tensor(np.stack((rows.flatten(), indices.flatten())))
        interp_mat = torch.sparse_coo_tensor(
            coords, torch.tensor(weights.flatten()), torch.Size((n_los, flat_size))
        )
        interp_mat = interp_mat.coalesce()
        return interp_mat

        return weights

    def get_lonlats(self) -> Tuple[np.ndarray, np.ndarray]:

        ll = self.center - 0.5 * self.extent[0] * self.x - 0.5 * self.extent[1] * self.y
        xx, yy = np.meshgrid(*map(np.arange, self.shape[:2]), indexing="ij")
        xx = xx * self.resolution[0]
        yy = yy * self.resolution[1]

        coords_ecef = ll[None, None] + self.x[None, None] * xx[..., None] + self.y[None, None] * yy[..., None]
        coords_lla = ecef_to_lla(coords_ecef)
        return coords_lla[..., 0], coords_lla[..., 1]

    def get_altitudes(self) -> np.ndarray:

        ll = self.center - 0.5 * self.extent[0] * self.x - 0.5 * self.extent[1] * self.y
        xx, yy, zz = np.meshgrid(*map(np.arange, self.shape), indexing="ij")
        xx = xx * self.resolution[0]
        yy = yy * self.resolution[1]
        zz = zz * self.resolution[2]

        coords_ecef = (
            ll[None, None]
            + self.x[None, None] * xx[..., None]
            + self.y[None, None] * yy[..., None]
            + self.z[None, None] * zz[..., None]
        )
        coords_lla = ecef_to_lla(coords_ecef)

        return coords_lla[..., -1]


    def interpolate_data(self, data: xr.Dataset) -> xr.Dataset:

        ll = self.center - 0.5 * self.extent[0] * self.x - 0.5 * self.extent[1] * self.y
        xx, yy, zz = np.meshgrid(*map(np.arange, self.shape), indexing="ij")
        xx = xx * self.resolution[0]
        yy = yy * self.resolution[1]
        zz = zz * self.resolution[2]

        coords_ecef = (
            ll[None, None]
            + self.x[None, None] * xx[..., None]
            + self.y[None, None] * yy[..., None]
            + self.z[None, None] * zz[..., None]
        )
        coords_lla = ecef_to_lla(coords_ecef)

        coords = xr.Dataset({
            "longitude": (("x", "y", "z"), coords_lla[..., 0]),
            "latitude": (("x", "y", "z"), coords_lla[..., 1]),
            "altitude": (("x", "y", "z"), coords_lla[..., 2])
        })

        atm_fields = data[[var for var in data if "altitude" in data[var].dims]]
        atm_fields = atm_fields.interp(
            latitude=coords.latitude,
            longitude=coords.longitude,
            altitude=coords.altitude,
        ).drop_vars(("latitude", "longitude", "altitude"))

        coords = xr.Dataset({
            "longitude": (("x", "y"), coords_lla[..., 0, 0]),
            "latitude": (("x", "y"), coords_lla[..., 0, 1]),
        })
        sfc_fields = data[[var for var in data if "altitude" not in data[var].dims]]
        sfc_fields = sfc_fields.interp(
            latitude=coords.latitude,
            longitude=coords.longitude,
        ).drop_vars(("latitude", "longitude"))
        return xr.merge((atm_fields, sfc_fields))



    def visualize(self):
        """
        Uses pyvista to display the atmosphere grid.
        """
        import pyvista as pv
        import numpy as np

        lll_ecef = self.center - 0.5 * self.extent[0] * self.x - 0.5 * self.extent[1] * self.y
        lll_ecef = self.center - 0.5 * self.extent[0] * self.x - 0.5 * self.extent[1] * self.y
        lrl_ecef = lll_ecef + self.extent[1] * self.y
        rrl_ecef = lrl_ecef + self.extent[0] * self.x
        rll_ecef = rrl_ecef - self.extent[1] * self.y

        llr_ecef = lll_ecef + self.extent[2] * self.z
        lrr_ecef = lrl_ecef + self.extent[2] * self.z
        rrr_ecef = rrl_ecef + self.extent[2] * self.z
        rlr_ecef = rll_ecef + self.extent[2] * self.z


        # Define the vertices of the cube (8 corners)
        vertices = np.array([
            lll_ecef,
            llr_ecef,
            lrl_ecef,
            lrr_ecef,
            rll_ecef,
            rlr_ecef,
            rrl_ecef,
            rrr_ecef,
        ])

        # Define the edges by specifying which vertices are connected
        edges = np.array([
        [0, 1], [1, 3], [3, 2], [2, 0],  # Bottom square
        [4, 5], [5, 7], [7, 6], [6, 4],  # Top square
        [0, 4], [1, 5], [2, 6], [3, 7]   # Vertical lines connecting top and bottom
        ])

        # Create a PyVista plotter
        plotter = pv.Plotter()

        # Add the cube edges as lines
        for edge in edges:
            line = pv.Line(vertices[edge[0]], vertices[edge[1]])
            plotter.add_mesh(line, color="black", line_width=3)

        xx, yy = np.meshgrid(self.xx, self.yy, indexing="ij")

        ll_ecef = self.center - 0.5 * self.extent[0] * self.x - 0.5 * self.extent[1] * self.y
        lr_ecef = ll_ecef + self.extent[1] * self.y
        rl_ecef = ll_ecef + self.extent[0] * self.x

        d_x = rl_ecef - ll_ecef
        xx = ll_ecef + d_x * self.shape[0]
        d_y = lr_ecef - ll_ecef
        yy = ll_ecef + d_y * self.shape[0]

        xx, yy = np.meshgrid(
            np.linspace(0, 1.0, self.shape[0] + 1),
            np.linspace(0, 1.0, self.shape[1] + 1),
            indexing="ij"
        )

        pts_ecef = ll_ecef[None, None] + xx[..., None] * d_x + yy[..., None] * d_y
        pts_lla = ecef_to_lla(pts_ecef)
        pts_lla[..., 2] = 0.0
        pts_ecef = lla_to_ecef(pts_lla)

        xx = pts_ecef[..., 0]
        yy = pts_ecef[..., 1]
        zz = pts_ecef[..., 2]
        surf = pv.StructuredGrid(xx, yy, zz)

        plotter.add_mesh(surf)

        # Add a grid for reference
        plotter.show_grid()


        # Show the plot
        plotter.show()


class LLAGrid:
    """
    A uniform latitude-longitude grid positioned on the surface of the Earth.
    """
    def __init__(
            self,
            center_lla: np.ndarray,
            resolution: Tuple[float],
            shape: Tuple[int],
    ):
        """
        Args:
            center_lla: The center of the grid in LLA coordinates.
            resolution: The resolution of the grid defined using a tuple
                of length three containing the longitude, latitude, and altitude
                resolution in units of degree, degree, and meter, respectively.
            shape: The shape of the grid.

        """
        self.center_lla = center_lla
        self.resolution = resolution
        self.shape = shape
        self.ll_lla = (
            self.center_lla
            - 0.5 * self.extent * np.array([1.0, 0.0, 0.0])
            - 0.5 * self.extent * np.array([0.0, 1.0, 0.0])
        )
        self.xx = self.ll_lla[0] + self.resolution[0] * np.arange(self.shape[0])
        self.yy = self.ll_lla[1] + self.resolution[1] * np.arange(self.shape[1])
        self.zz = self.ll_lla[2] + self.resolution[2] * np.arange(self.shape[2])


    @property
    def extent(self) -> np.ndarray:
        """
        A length-3 vector containing the dimensions of the grid in x, y, and z directions.
        """
        return self.resolution * self.shape


    def ecef_to_native(self, coords_ecef: np.ndarray) -> np.ndarray:
        """
        Convert ECEF coordinates to native coordinates of the grid.

        Args:
            coords_ecef: An array containing ECEF coordinates oriented along the last dimension.

        Return:
            An array of similar shape as 'coords_ecef' containing the coordinates converted
            to native grid coordinates.
        """
        coords_lla = ecef_to_lla(coords_ecef)
        translated = coords_lla - self.ll_lla
        x = translated[..., 0] / self.resolution[0]
        y = translated[..., 1] / self.resolution[1]
        z = translated[..., 2] / self.resolution[2]
        return np.stack((x, y, z), -1)


    def calculate_path_integral_matrices(
            self,
            origin_ecef: np.ndarray,
            los_ecef: np.ndarray,
            step_length: float,
            n_steps: int,
            resolution: int = 100
    ) -> np.ndarray:
        """
        Calculates matrices that calculate path integrals through the atmospheric field.

        Args:
            origin_ecef: A [n_los, 3] numpy.ndarray defining the origins of the lines of sight.
            los_ecef: A [n_los, 3] numpy.ndarray defining the lines of sight.
            step_length: The length of the steps along the path.
            n_steps: The number of steps.
            resolution: The number of quadrature step to apply to each step along the pencil beam.

        Return:
            A numpy.ndarray of shape [n_los, n_steps, atm_size] containing the matrices to extract
            path integrals from flattened atmospheric fields.
        """

        origin_ecef = origin_ecef.reshape(-1, origin_ecef.shape[-1])
        los_ecef = los_ecef.reshape(-1, los_ecef.shape[-1])
        n_los = origin_ecef.shape[0]

        los_ecef = los_ecef.astype("float64")
        los_ecef = los_ecef / np.linalg.norm(los_ecef, axis=-1, keepdims=True) * step_length / resolution

        curr_pos = origin_ecef

        max_inds = np.array([shp - 2 for shp in self.shape])
        flat_size = np.prod(self.shape)

        step_matrices = []

        for step in tqdm(range(n_steps)):

            step_indices = np.zeros((n_los, resolution + 1, 8), np.int64)
            step_weights = np.zeros((n_los, resolution + 1, 8), np.float32)

            for int_step in range(resolution + 1):

                curr_pos_native = self.ecef_to_native(curr_pos)
                lll = np.maximum(np.minimum(np.trunc(curr_pos_native).astype(np.int64), max_inds), 0)
                bcoords = np.clip(curr_pos_native - lll, 0, 1)

                lll_x = lll[..., 0]
                lll_y = lll[..., 1]
                lll_z = lll[..., 2]

                weight_ind  = 0
                for offset_x in range(2):
                    for offset_y in range(2):
                        for offset_z in range(2):
                            inds = np.ravel_multi_index(
                                (lll_x + offset_x, lll_y + offset_y, lll_z + offset_z),
                                self.shape
                            )
                            weights = (
                                (1.0 - bcoords[..., 0] - offset_x + 2.0 * offset_x * bcoords[..., 0]) *
                                (1.0 - bcoords[..., 1] - offset_y + 2.0 * offset_y * bcoords[..., 1]) *
                                (1.0 - bcoords[..., 2] - offset_z + 2.0 * offset_z * bcoords[..., 2])
                            )
                            step_indices[:, int_step, weight_ind] = inds
                            if int_step == 0 or int_step == resolution:
                                step_weights[:, int_step, weight_ind] = 0.5 * weights / resolution
                            else:
                                step_weights[:, int_step, weight_ind] = weights / resolution
                            weight_ind += 1

                curr_pos += los_ecef

            rows = np.broadcast_to(np.arange(n_los)[:, None, None], step_indices.shape)
            coords = torch.tensor(np.stack((rows.flatten(), step_indices.flatten())))
            step_mat = torch.sparse_coo_tensor(
                coords, torch.tensor(step_weights.flatten()), torch.Size((n_los, flat_size))
            )
            step_mat = step_mat.coalesce()
            step_matrices.append(step_mat)

            curr_pos -= los_ecef

        return step_matrices


    def calculate_surface_interpolation_matrices(
            self,
            origin_ecef: np.ndarray,
    ) -> np.ndarray:
        """
        Calculates matrices that extract surface values from the surface fields.

        Args:
            origin_ecef: A [n_los, 3] numpy.ndarray defining the origins of the lines of sight.

        Return:
            A numpy.ndarray of shape [n_los, atm_flat] that can be used to calculate
        """
        origin_ecef = origin_ecef.reshape(-1, origin_ecef.shape[-1])
        n_los = origin_ecef.shape[0]
        origin_native = self.ecef_to_native(origin_ecef)

        max_inds = np.array([shp - 2 for shp in self.shape])
        flat_size = np.prod(self.shape[:2])

        lll = np.maximum(np.minimum(np.trunc(origin_native).astype(np.int64), max_inds), 0)
        bcoords = np.clip(origin_native - lll, 0, 1)

        weights = np.zeros((n_los, 4), dtype="float32")
        indices = np.zeros((n_los, 4), dtype=np.int64)

        lll_x = lll[..., 0]
        lll_y = lll[..., 1]
        lll_z = lll[..., 2]

        weight_ind = 0
        for offset_x in range(2):
            for offset_y in range(2):
                inds = np.ravel_multi_index(
                    (lll_x + offset_x, lll_y + offset_y),
                    self.shape[:2]
                )
                indices[:, weight_ind] = inds
                interp_weights = (
                    (1.0 - bcoords[..., 0] - offset_x + 2.0 * offset_x * bcoords[..., 0]) *
                    (1.0 - bcoords[..., 1] - offset_y + 2.0 * offset_y * bcoords[..., 1])
                )
                weights[:, weight_ind] = interp_weights
                weight_ind += 1

        rows = np.broadcast_to(np.arange(n_los)[:, None], indices.shape)
        coords = torch.tensor(np.stack((rows.flatten(), indices.flatten())))
        interp_mat = torch.sparse_coo_tensor(
            coords, torch.tensor(weights.flatten()), torch.Size((n_los, flat_size))
        )
        interp_mat = interp_mat.coalesce()
        return interp_mat

    def get_lonlats(self) -> Tuple[np.ndarray, np.ndarray]:

        return np.meshgrid(self.xx, self.yy)

    def get_altitudes(self) -> np.ndarray:

        alt = np.broadcast_to(self.zz[None, None], self.shape)
        return alt


    def interpolate_data(self, data: xr.Dataset) -> xr.Dataset:
        return data.interp(
            longitude=self.xx,
            latitude=self.yy,
            altitude=self.zz
        )

    def visualize(self):
        """
        Uses pyvista to display the atmosphere grid.
        """
        import pyvista as pv
        import numpy as np

        xx, yy, zz = np.meshgrid(self.x, self.y, self.z, indexing="ij")


        x =  np.array([1.0, 0.0, 0.0])
        y =  np.array([0.0, 1.0, 0.0])
        z =  np.array([0.0, 0.0, 1.0])

        lll_lla = self.center_lla - 0.5 * self.extent[0] * x - 0.5 * self.extent[1] * y
        lrl_lla = lll_lla + self.extent[1] * x
        rrl_lla = lrl_lla + self.extent[0] * y
        rll_lla = rrl_lla - self.extent[1] * x
        lll_ecef = lla_to_ecef(lll_lla)
        lrl_ecef = lla_to_ecef(lrl_lla)
        rrl_ecef = lla_to_ecef(rll_lla)
        rll_ecef = lla_to_ecef(rrl_lla)

        llr_lla = lll_lla + self.extent[2] * z
        lrr_lla = lrl_lla + self.extent[2] * z
        rrr_lla = rrl_lla + self.extent[2] * z
        rlr_lla = rll_lla + self.extent[2] * z
        llr_ecef = lla_to_ecef(llr_lla)
        lrr_ecef = lla_to_ecef(lrr_lla)
        rrr_ecef = lla_to_ecef(rlr_lla)
        rlr_ecef = lla_to_ecef(rrr_lla)


        # Define the vertices of the cube (8 corners)
        vertices = np.array([
            lll_ecef,
            llr_ecef,
            lrl_ecef,
            lrr_ecef,
            rll_ecef,
            rlr_ecef,
            rrl_ecef,
            rrr_ecef,
        ])

        # Define the edges by specifying which vertices are connected
        edges = np.array([
            [0, 2], [2, 4], [4, 6], [6, 0],  # Bottom
            [1, 3], [3, 5], [5, 7], [7, 1],  # Top
            [0, 1], [2, 3], [4, 5], [6, 7],  # Sides
        ])

        # Create a PyVista plotter
        plotter = pv.Plotter()

        # Add the cube edges as lines
        for edge in edges:
            line = pv.Line(vertices[edge[0]], vertices[edge[1]])
            plotter.add_mesh(line, color="black", line_width=3)


        xx, yy, zz = np.meshgrid(self.xx, self.yy, self.zz, indexing="ij")
        pts_ecef = lla_to_ecef(np.stack((xx, yy, zz), -1))
        xx = pts_ecef[..., 0]
        yy = pts_ecef[..., 1]
        zz = pts_ecef[..., 2]

        surf = pv.StructuredGrid(xx, yy, zz)

        #plotter.add_mesh(surf)

        # Add a grid for reference
        plotter.show_grid()


        # Show the plot
        plotter.show()
