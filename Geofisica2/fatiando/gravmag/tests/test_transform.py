from __future__ import division, absolute_import
import pytest
import numpy as np
import numpy.testing as npt
from .. import transform, prism
from ... import gridder, utils
from ...mesher import Prism


def _trim(array, shape, d=20):
    "Remove d elements from the edges of an array"
    return array.reshape(shape)[d: -d, d: -d].ravel()


def test_pole_reduce():
    "gravmag.transform pole reduction matches analytical solution"
    # Use remanent magnetization
    sinc, sdec = -70, 30
    model = [Prism(-100, 100, -500, 500, 0, 100,
                   {'density': 1000,
                    'magnetization': utils.ang2vec(5, sinc, sdec)})]
    # Use low latitudes to make sure that there are no problems with FFT
    # instability.
    inc, dec = -60, -15
    shape = (50, 50)
    x, y, z = gridder.regular([-2000, 2000, -2000, 2000], shape, z=-100)
    data = prism.tf(x, y, z, model, inc, dec)
    pole = transform.reduce_to_pole(x, y, data, shape, inc, dec, sinc, sdec)
    pole_true = prism.tf(x, y, z, model, -90, 0, pmag=utils.ang2vec(5, -90, 0))
    npt.assert_allclose(pole, pole_true, atol=10, rtol=0.01)


def test_upcontinue():
    "gravmag.transform upward continuation matches analytical solution"
    model = [Prism(-1000, 1000, -500, 500, 0, 1000,
                   {'density': 1000,
                    'magnetization': utils.ang2vec(5, 20, -30)})]
    shape = (100, 100)
    inc, dec = -10, 15
    x, y, z = gridder.regular([-5000, 5000, -5000, 5000], shape, z=-500)
    dz = 10
    fields = 'potential gx gy gz gxx gxy gxz gyy gyz gzz'.split()
    accuracy = [0.002, 0.2, 0.2, 0.3, 2, 2, 4, 4, 4, 6]
    for f, atol in zip(fields, accuracy):
        func = getattr(prism, f)
        data = func(x, y, z, model)
        analytical = func(x, y, z + dz, model)
        up = transform.upcontinue(x, y, data, shape, dz)
        diff = np.abs(up - analytical)
        check = diff <= atol
        assert np.all(check), \
            'Failed for {} (mismatch {:.2f}%)'.format(
            f, 100*(check.size - check.sum())/check.size)
    data = prism.tf(x, y, z, model, inc, dec)
    analytical = prism.tf(x, y, z + dz, model, inc, dec)
    up = transform.upcontinue(x, y, data, shape, dz)
    diff = np.abs(up - analytical)
    check = diff <= 15
    assert np.all(check), \
        'Failed for tf (mismatch {:.2f}%)'.format(
        100*(check.size - check.sum())/check.size)


def test_upcontinue_warning():
    "gravmag.transform upward continuation raises warning if height <= 0"
    model = [Prism(-1000, 1000, -500, 500, 0, 1000, {'density': 1000})]
    shape = (100, 100)
    x, y, z = gridder.regular([-5000, 5000, -5000, 5000], shape, z=-500)
    data = prism.gz(x, y, z, model)
    with pytest.warns(UserWarning):
        transform.upcontinue(x, y, data, shape, height=0)
    with pytest.warns(UserWarning):
        transform.upcontinue(x, y, data, shape, height=-100)


def test_second_horizontal_derivatives_fd():
    "gravmag.transform 2nd xy derivatives by finite diff against analytical"
    model = [Prism(-1000, 1000, -500, 500, 0, 2000, {'density': 100})]
    shape = (300, 300)
    x, y, z = gridder.regular([-10000, 10000, -10000, 10000], shape, z=-500)
    derivatives = 'xx yy'.split()
    grav = prism.potential(x, y, z, model)
    for deriv in derivatives:
        analytical = getattr(prism, 'g{}'.format(deriv))(x, y, z, model)
        func = getattr(transform, 'deriv' + deriv[0])
        calculated = utils.si2eotvos(func(x, y, grav, shape, method='fd',
                                          order=2))
        diff = np.abs(analytical - calculated)
        assert np.all(diff/np.abs(analytical).max() <= 0.01), \
            "Failed for g{}. Max: {} Mean: {} STD: {}".format(
                deriv, diff.max(), diff.mean(), diff.std())


def test_horizontal_derivatives_fd():
    "gravmag.transform 1st xy derivatives by finite diff against analytical"
    model = [Prism(-1000, 1000, -500, 500, 0, 2000, {'density': 100})]
    shape = (300, 300)
    x, y, z = gridder.regular([-5000, 5000, -5000, 5000], shape, z=-200)
    derivatives = 'x y'.split()
    grav = utils.mgal2si(prism.gz(x, y, z, model))
    for deriv in derivatives:
        analytical = getattr(prism, 'g{}z'.format(deriv))(x, y, z, model)
        func = getattr(transform, 'deriv' + deriv)
        calculated = utils.si2eotvos(func(x, y, grav, shape, method='fd'))
        diff = np.abs(analytical - calculated)
        assert np.all(diff <= 0.005*np.abs(analytical).max()), \
            "Failed for g{}. Max: {} Mean: {} STD: {}".format(
                deriv, diff.max(), diff.mean(), diff.std())


def test_derivatives_uneven_shape():
    "gravmag.transform FFT derivatives work if grid spacing is uneven"
    model = [Prism(-1000, 1000, -500, 500, 0, 2000, {'density': 100})]
    shape = (150, 300)
    x, y, z = gridder.regular([-10000, 10000, -10000, 10000], shape, z=-100)
    grav = utils.mgal2si(prism.gz(x, y, z, model))
    analytical = prism.gzz(x, y, z, model)
    calculated = utils.si2eotvos(transform.derivz(x, y, grav, shape,
                                                  method='fft'))
    diff = _trim(np.abs(analytical - calculated), shape)
    assert np.all(diff <= 0.005*np.abs(analytical).max()), \
        "Failed for gzz"


def test_gz_derivatives():
    "gravmag.transform FFT 1st derivatives of gz against analytical solutions"
    model = [Prism(-1000, 1000, -500, 500, 0, 2000, {'density': 100})]
    shape = (300, 300)
    x, y, z = gridder.regular([-10000, 10000, -10000, 10000], shape, z=-100)
    derivatives = 'x y z'.split()
    grav = utils.mgal2si(prism.gz(x, y, z, model))
    for deriv in derivatives:
        analytical = getattr(prism, 'g{}z'.format(deriv))(x, y, z, model)
        calculated = utils.si2eotvos(
            getattr(transform, 'deriv' + deriv)(x, y, grav, shape,
                                                method='fft'))
        diff = _trim(np.abs(analytical - calculated), shape)
        assert np.all(diff <= 0.005*np.abs(analytical).max()), \
            "Failed for g{}z".format(deriv)


def test_gx_derivatives():
    "gravmag.transform FFT 1st derivatives of gx against analytical solutions"
    model = [Prism(-1000, 1000, -500, 500, 0, 2000, {'density': 100})]
    shape = (300, 300)
    x, y, z = gridder.regular([-10000, 10000, -10000, 10000], shape, z=-100)
    derivatives = 'x y z'.split()
    grav = utils.mgal2si(prism.gx(x, y, z, model))
    for deriv in derivatives:
        analytical = getattr(prism, 'gx{}'.format(deriv))(x, y, z, model)
        calculated = utils.si2eotvos(
            getattr(transform, 'deriv' + deriv)(x, y, grav, shape,
                                                method='fft'))
        diff = _trim(np.abs(analytical - calculated), shape)
        assert np.all(diff <= 0.005*np.abs(analytical).max()), \
            "Failed for gx{}".format(deriv)


def test_gy_derivatives():
    "gravmag.transform FFT 1st derivatives of gy against analytical solutions"
    model = [Prism(-1000, 1000, -500, 500, 0, 2000, {'density': 100})]
    shape = (300, 300)
    x, y, z = gridder.regular([-10000, 10000, -10000, 10000], shape, z=-100)
    derivatives = 'x y z'.split()
    grav = utils.mgal2si(prism.gy(x, y, z, model))
    for deriv in derivatives:
        if deriv == 'x':
            func = getattr(prism, 'g{}y'.format(deriv))
        else:
            func = getattr(prism, 'gy{}'.format(deriv))
        analytical = func(x, y, z, model)
        calculated = utils.si2eotvos(
            getattr(transform, 'deriv' + deriv)(x, y, grav, shape,
                                                method='fft'))
        diff = _trim(np.abs(analytical - calculated), shape)
        assert np.all(diff <= 0.005*np.abs(analytical).max()), \
            "Failed for gy{}".format(deriv)


def test_second_derivatives():
    "gravmag.transform FFT second derivatives against analytical solutions"
    model = [Prism(-1000, 1000, -500, 500, 0, 2000, {'density': -200})]
    shape = (300, 300)
    x, y, z = gridder.regular([-10000, 10000, -10000, 10000], shape, z=-100)
    derivatives = 'xx yy zz'.split()
    pot = prism.potential(x, y, z, model)
    for deriv in derivatives:
        analytical = getattr(prism, 'g{}'.format(deriv))(x, y, z, model)
        calculated = utils.si2eotvos(
            getattr(transform, 'deriv' + deriv[0])(x, y, pot, shape, order=2,
                                                   method='fft'))
        diff = _trim(np.abs(analytical - calculated), shape)
        assert np.all(diff <= 0.005*np.abs(analytical).max()), \
            "Failed for g{}. Max: {} Mean: {} STD: {}".format(
                deriv, diff.max(), diff.mean(), diff.std())


def test_laplace_from_potential():
    "gravmag.transform 2nd derivatives of potential obey the Laplace equation"
    model = [Prism(-1000, 1000, -500, 500, 0, 2000, {'density': 200})]
    shape = (300, 300)
    x, y, z = gridder.regular([-10000, 10000, -10000, 10000], shape, z=-100)
    potential = prism.potential(x, y, z, model)
    gxx = utils.si2eotvos(transform.derivx(x, y, potential, shape, order=2,
                                           method='fft'))
    gyy = utils.si2eotvos(transform.derivy(x, y, potential, shape, order=2,
                                           method='fft'))
    gzz = utils.si2eotvos(transform.derivz(x, y, potential, shape, order=2))
    laplace = _trim(gxx + gyy + gzz, shape)
    assert np.all(np.abs(laplace) <= 1e-10), \
        "Max: {} Mean: {} STD: {}".format(
            laplace.max(), laplace.mean(), laplace.std())


def test_tilt_sane_values():
    "gravmag.transform tilt returns sane values, between -90 and 90 degrees"
    inc, dec = 90, 0
    mag = utils.ang2vec(200, inc, dec)
    model = [Prism(-500, 500, -500, 500, 0, 2000, {'magnetization': mag})]
    shape = (300, 300)
    x, y, z = gridder.regular([-10000, 10000, -10000, 10000], shape, z=-100)
    data = prism.tf(x, y, z, model, inc, dec)
    tilt = np.degrees(transform.tilt(x, y, data, shape))
    assert tilt.max() < 90, \
        "Maximum tilt greater than 90: {}".format(tilt.max())
    assert tilt.min > -90, \
        "Minimum tilt less than -90: {}".format(tilt.min())


def test_tilt_analytical_derivatives():
    "gravmag.transform tilt returns same values given analytical derivatives"
    model = [Prism(-100, 100, -100, 100, 0, 100, {'density': 1000})]
    shape = (400, 400)
    x, y, z = gridder.regular([-10000, 10000, -10000, 10000], shape, z=-100)
    data = utils.mgal2si(prism.gz(x, y, z, model))
    dx = utils.eotvos2si(prism.gxz(x, y, z, model))
    dy = utils.eotvos2si(prism.gyz(x, y, z, model))
    dz = utils.eotvos2si(prism.gzz(x, y, z, model))
    tilt_analytical = transform.tilt(x, y, data, shape, dx, dy, dz)
    tilt_numerical = transform.tilt(x, y, data, shape)
    npt.assert_allclose(tilt_numerical, tilt_analytical, rtol=0.10)


def test_radial_average_spectrum_distances():
    shape = (201, 201)
    area = (-100, 100, -100, 100)
    x, y = gridder.regular(area, shape)
    x, y = x.reshape(shape), y.reshape(shape)
    x, y = np.fft.ifftshift(x), np.fft.ifftshift(y)
    distances = np.sqrt(x**2 + y**2)
    k, radial_distances = transform.radial_average_spectrum(x, y, distances)
    npt.assert_allclose(k[2:], radial_distances[2:], rtol=0.1)  # doesn't fail


def test_radial_average_spectrum_integers():
    x = np.arange(-10, 11, 1)
    y = np.arange(-10, 11, 1)
    y, x = np.meshgrid(y, x)
    x, y = np.fft.ifftshift(x), np.fft.ifftshift(y)
    r = np.sqrt(x**2 + y**2)
    z = np.zeros(x.shape)
    integers = np.arange(0, x.max() + 1, 1)
    for i in integers:
        if i == 0:
            inside = r <= 0.5
        else:
            inside = np.logical_and(r > i - 0.5, r <= i + 0.5)
        z[inside] = i
    k, radial_z = transform.radial_average_spectrum(x, y, z)
    npt.assert_allclose(integers, radial_z, rtol=0.1)
