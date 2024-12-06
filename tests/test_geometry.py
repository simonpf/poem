"""
Tests for the poem.geometry module
"""
import numpy as np
import torch

from poem.geometry import (
    lla_to_ecef,
    ecef_to_lla,
    SEM_A,
    SEM_B,
    calculate_surface_intersection,
    EuclideanGrid,
    LLAGrid
)


def test_lla_to_ecef():
    """
    Test that conversion from LLA to ECEF coordinates works.
    """
    coords_lla = np.array([0, 0, 0])
    coords_ecef = lla_to_ecef(coords_lla)
    assert np.isclose(coords_ecef[0], SEM_A)
    assert np.isclose(coords_ecef[1], 0.0)
    assert np.isclose(coords_ecef[2], 0.0)

    coords_lla = np.array([90, 0, 0])
    coords_ecef = lla_to_ecef(coords_lla)
    assert np.isclose(coords_ecef[0], 0.0)
    assert np.isclose(coords_ecef[1], SEM_A)
    assert np.isclose(coords_ecef[2], 0.0)

    coords_lla = np.array([0, 90, 0])
    coords_ecef = lla_to_ecef(coords_lla)
    assert np.isclose(coords_ecef[0], 0.0)
    assert np.isclose(coords_ecef[1], 0.0)
    assert np.isclose(coords_ecef[2], SEM_B)


def test_ecef_to_lla():
    """
    Test that conversion from ECEF to LLA  works.
    """
    coords_ecef = np.array([SEM_A, 0, 0])
    coords_lla = ecef_to_lla(coords_ecef)
    assert np.isclose(coords_lla[0], 0.0)
    assert np.isclose(coords_lla[1], 0.0)
    assert np.isclose(coords_lla[2], 0.0)

    coords_ecef = np.array([0, SEM_A, 0])
    coords_lla = ecef_to_lla(coords_ecef)
    assert np.isclose(coords_lla[0], 90.0)
    assert np.isclose(coords_lla[1], 0.0)
    assert np.isclose(coords_lla[2], 0.0)

    coords_ecef = np.array([0, 0, SEM_B])
    coords_lla = ecef_to_lla(coords_ecef)
    assert np.isclose(coords_lla[0], 0.0)
    assert np.isclose(coords_lla[1], 90.0)
    assert np.isclose(coords_lla[2], 0.0)


def test_calculate_surface_intersection(

):
    """
    Test calculate of intersection with Earth surface using sensor
    position, line-of-sight, and footprint position of actual AMSR2
    observations.
    """
    sensor_pos_ecef = np.array([-6595187.5 , -2491061.  ,   673912.94])
    sensor_los_ecef = np.array([1004408.  , -433672.75,  255562.06])
    fp_pos_ecef = calculate_surface_intersection(sensor_pos_ecef, sensor_los_ecef)
    fp_pos_lla = ecef_to_lla(fp_pos_ecef)

    assert np.isclose(fp_pos_lla[0], -152.38435)
    assert np.isclose(fp_pos_lla[1], 8.435728)
    assert np.isclose(fp_pos_lla[2], 0.0)


def test_euclidean_native_coordinates():
    """
    Test conversion to native coordinates of atmosphere grid.
    """
    center_lla = np.array([0.0, 0.0, 0.0])
    center_ecef = lla_to_ecef(center_lla)
    north_ecef = lla_to_ecef(np.array([0.0, 1.0, 0.0]))
    orientation = north_ecef - center_ecef
    atm = EuclideanGrid(
        center_ecef=center_ecef,
        orientation_ecef=orientation,
        resolution=np.array([1.0e3, 2.0e3, 0.5e3]),
        shape=(100, 100, 100)
    )

    xx = atm.xx
    xx_native = atm.ecef_to_native(xx)
    assert np.all(np.isclose(xx_native[..., 0], np.arange(atm.shape[0])))

    yy = atm.yy
    yy_native = atm.ecef_to_native(yy)
    assert np.all(np.isclose(yy_native[..., 1], np.arange(atm.shape[1])))

    zz = atm.zz
    zz_native = atm.ecef_to_native(zz)
    assert np.all(np.isclose(zz_native[..., 2], np.arange(atm.shape[2])))


def test_calculate_path_integral_matrices_euclidean():
    """
    Test calculation of the path integration matrices for atmospheric fields for
    Euclidean grids.
    """
    center_lla = np.array([0.0, 0.0, 0.0])
    center_ecef = lla_to_ecef(center_lla)
    north_ecef = lla_to_ecef(np.array([0.0, 1.0, 0.0]))
    orientation = north_ecef - center_ecef
    atm = EuclideanGrid(
        center_ecef=center_ecef,
        orientation_ecef=orientation,
        resolution=np.array([3.0e3, 2.0e3, 1.0e3]),
        shape=(10, 10, 10)
    )

    interp_matrix = atm.calculate_path_integral_matrices(
        atm.center,
        np.array([1.0, 0.0, 0.0]),
        step_length = 1.0e3,
        n_steps = 10,
        resolution=100
    )

    xx, yy, zz = np.meshgrid(*list(map(np.arange, atm.shape)), indexing="ij")

    zz = torch.tensor(zz.flatten().astype(np.float32))
    for ind in range(9):
        layer_alt = interp_matrix[ind] @ zz
        assert np.all(np.isclose(layer_alt[0].item(), ind + 0.5))

    interp_matrix = atm.calculate_path_integral_matrices(
        atm.xx[0],
        np.array([0.0, 0.0, 1.0]),
        step_length = 3.0e3,
        n_steps = 10
    )
    xx = torch.tensor(xx.flatten().astype(np.float32))
    for ind in range(9):
        mean = interp_matrix[ind] @ xx
        assert np.all(np.isclose(mean[0].item(), ind + 0.5))

    interp_matrix = atm.calculate_path_integral_matrices(
        atm.yy[0],
        np.array([0.0, 1.0, 0.0]),
        step_length = 2.0e3,
        n_steps = 10
    )
    yy = torch.tensor(yy.flatten().astype(np.float32))
    for ind in range(9):
        mean = interp_matrix[ind] @ yy
        assert np.all(np.isclose(mean[0].item(), ind + 0.5))


def test_calculate_surface_interpolation_matrices_euclidean():
    """
    Test calculation of the surface interpolation matrices for Euclidean grids.
    """
    center_lla = np.array([0.0, 0.0, 0.0])
    center_ecef = lla_to_ecef(center_lla)
    north_ecef = lla_to_ecef(np.array([0.0, 1.0, 0.0]))
    orientation = north_ecef - center_ecef
    atm = EuclideanGrid(
        center_ecef=center_ecef,
        orientation_ecef=orientation,
        resolution=np.array([3.0e3, 2.0e3, 1.0e3]),
        shape=(11, 11, 11)
    )

    interp_matrix = atm.calculate_surface_interpolation_matrices(atm.center)

    xx, yy = np.meshgrid(*list(map(np.arange, atm.shape[:2])), indexing="ij")

    xx = torch.tensor(xx.flatten().astype("float32"))
    cntr = interp_matrix @ xx
    assert np.isclose(cntr[0].item(), 5.5)

    yy = torch.tensor(yy.flatten().astype("float32"))
    cntr = interp_matrix @ yy
    assert np.isclose(cntr[0], 5.5)


def test_lla_native_coordinates():
    """
    Test conversion to native coordinates for LLA grids.
    """
    center_lla = np.array([0.0, 0.0, 0.0])
    atm = LLAGrid(
        center_lla=center_lla,
        resolution=np.array([0.01, 0.01, 1e3]),
        shape=(101, 101, 101)
    )

    xx = atm.xx
    yy = atm.yy
    zz = atm.zz
    xx, yy, zz = np.meshgrid(xx, yy, zz, indexing="ij")
    coords_ecef = lla_to_ecef(np.stack((xx, yy, zz), -1))

    xx_native = atm.ecef_to_native(coords_ecef[:, 0, 0])
    assert np.all(np.isclose(xx_native[..., 0], np.arange(atm.shape[0]), rtol=1e-2))

    yy_native = atm.ecef_to_native(coords_ecef[0, :, 0])
    assert np.all(np.isclose(yy_native[..., 1], np.arange(atm.shape[1]), rtol=1e-2))

    zz_native = atm.ecef_to_native(coords_ecef[0, 0, :])
    assert np.all(np.isclose(zz_native[..., 2], np.arange(atm.shape[2]), rtol=1e-1))



def test_calculate_path_integral_matrices_lla():
    """
    Test calculation of the path integration matrices for atmospheric fields.
    """
    center_lla = np.array([0.0, 0.0, 0.0])
    atm = LLAGrid(
        center_lla=center_lla,
        resolution=np.array([0.01, 0.01, 1e3]),
        shape=(10, 10, 10)
    )

    interp_matrix = atm.calculate_path_integral_matrices(
        lla_to_ecef(atm.center_lla),
        np.array([1.0, 0.0, 0.0]),
        step_length = 1.0e3,
        n_steps = 10,
        resolution=100
    )

    xx, yy, zz = np.meshgrid(atm.xx, atm.yy, atm.zz, indexing="ij")
    coords_ecef = lla_to_ecef(np.stack((xx, yy, zz), -1))

    xx, yy, zz = np.meshgrid(*list(map(np.arange, atm.shape)), indexing="ij")
    zz = torch.tensor(zz.flatten().astype(np.float32))
    for ind in range(9):
        layer_alt = interp_matrix[ind] @ zz
        assert np.all(np.isclose(layer_alt[0], ind + 0.5))

    los_lon = coords_ecef[1, 0, 0] - coords_ecef[0, 0, 0]
    interp_matrix = atm.calculate_path_integral_matrices(
        coords_ecef[:1, 0, 0],
        los_lon,
        step_length = 1.11e3,
        n_steps = 10
    )
    xx = torch.tensor(xx.flatten().astype(np.float32))
    for ind in range(9):
        mean = interp_matrix[ind] @ xx
        assert np.all(np.isclose(mean[0], ind + 0.5, rtol=1e-1))

    los_lat = coords_ecef[5, 1, 0] - coords_ecef[5, 0, 0]
    interp_matrix = atm.calculate_path_integral_matrices(
        coords_ecef[:1, 0, 0],
        los_lat,
        step_length = 1.11e3,
        n_steps = 10
    )
    yy = torch.tensor(yy.flatten().astype(np.float32))
    for ind in range(9):
        mean = interp_matrix[ind] @ yy
        assert np.all(np.isclose(mean[0], ind + 0.5, rtol=1e-1))


def test_calculate_surface_interpolation_matrices_lla():
    """
    Test calculation of the surface interpolation matrices LLA grids.
    """
    center_lla = np.array([0.0, 0.0, 0.0])
    atm = LLAGrid(
        center_lla=center_lla,
        resolution=np.array([0.1, 0.1, 1e3]),
        shape=(11, 11, 11)
    )

    interp_matrix = atm.calculate_surface_interpolation_matrices(atm.center_lla)

    xx, yy = np.meshgrid(*list(map(np.arange, atm.shape[:2])), indexing="ij")

    xx = torch.tensor(xx.flatten().astype("float32"))
    cntr = interp_matrix @ xx
    assert np.isclose(cntr[0].item(), 5.5)

    yy = torch.tensor(yy.flatten().astype("float32"))
    cntr = interp_matrix @ yy
    assert np.isclose(cntr[0].item(), 5.5)
