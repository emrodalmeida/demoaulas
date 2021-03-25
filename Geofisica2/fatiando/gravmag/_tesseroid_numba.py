"""
A numba implementation of the tesseroid gravity effects.

These functions compute the effect of a single tesseroid. They are used by
fatiando.gravmag.tesseroid as a backend and are not meant to be used directly.

A few doctests for the numba code::

>>> import numpy as np
>>> stack = np.empty((6, 6))
>>> stktop = -1
>>> stktop = split(0, 4, 3, 6, 11, 5, 2, 1, 3, stack, stktop)
>>> stktop
5
>>> stack
array([[  0.,   2.,   3.,   6.,   7.,   5.],
       [  0.,   2.,   3.,   6.,   9.,   7.],
       [  0.,   2.,   3.,   6.,  11.,   9.],
       [  2.,   4.,   3.,   6.,   7.,   5.],
       [  2.,   4.,   3.,   6.,   9.,   7.],
       [  2.,   4.,   3.,   6.,  11.,   9.]])

"""
from __future__ import division, absolute_import
import numba
import numpy as np

from ..constants import MEAN_EARTH_RADIUS


nodes = np.array([-0.577350269189625731058868041146,
                  0.577350269189625731058868041146])


def engine_factory(kernel):
    """
    Make the engine functions for each specific field by passing in the
    appropriate kernel.
    """
    @numba.jit(nopython=True)
    def engine(lon, sinlat, coslat, radius, bounds, density, ratio,
               stack, lonc, sinlatc, coslatc, rc, result):
        error_code = 0
        for l in range(result.size):
            for i in range(6):
                stack[0, i] = bounds[i]
            stktop = 0
            while stktop >= 0:
                w, e, s, n, top, bottom = stack[stktop, :]
                stktop -= 1
                distance, Llon, Llat, Lr = distance_size(
                    lon[l], coslat[l], sinlat[l], radius[l], w, e, s, n, top,
                    bottom)
                nlon, nlat, nr, new_cells, err = divisions(
                    distance, Llon, Llat, Lr, ratio)
                error_code += err
                if new_cells > 1:
                    if new_cells + (stktop + 1) > stack.shape[0]:
                        raise OverflowError
                    stktop = split(w, e, s, n, top, bottom, nlon, nlat, nr,
                                   stack, stktop)
                else:
                    scale = scale_nodes(w, e, s, n, top, bottom, nodes, lonc,
                                        sinlatc, coslatc, rc)
                    result[l] += density*scale*kernel(
                        lon[l], coslat[l], sinlat[l], radius[l], lonc, sinlatc,
                        coslatc, rc)
        return error_code
    return engine


@numba.jit(nopython=True)
def scale_nodes(w, e, s, n, top, bottom, nodes, lonc, sinlatc, coslatc, rc):
    "Put the GLQ nodes in the integration limit"
    d2r = np.pi/180
    dlon = d2r*(e - w)
    dlat = d2r*(n - s)
    dr = top - bottom
    # Scale the GLQ nodes to the integration limits
    for i in range(len(nodes)):
        lonc[i] = 0.5*dlon*nodes[i] + d2r*0.5*(e + w)
        latc = 0.5*dlat*nodes[i] + d2r*0.5*(n + s)
        sinlatc[i] = np.sin(latc)
        coslatc[i] = np.cos(latc)
        rc[i] = (0.5*dr*nodes[i] +
                 0.5*(top + bottom) + MEAN_EARTH_RADIUS)
    scale = dlon*dlat*dr*0.125
    return scale


@numba.jit(nopython=True)
def distance_size(lon, coslat, sinlat, radius, w, e, s, n, top, bottom):
    "Calculate the distance to the center of the tesseroid and its dimensions"
    d2r = np.pi/180
    rt = 0.5*(top + bottom) + MEAN_EARTH_RADIUS
    lont = d2r*0.5*(w + e)
    latt = d2r*0.5*(s + n)
    sinlatt = np.sin(latt)
    coslatt = np.cos(latt)
    cospsi = sinlat*sinlatt + coslat*coslatt*np.cos(lon - lont)
    distance = np.sqrt(radius**2 + rt**2 - 2*radius*rt*cospsi)
    # Calculate the dimensions of the tesseroid in meters
    rtop = top + MEAN_EARTH_RADIUS
    Llon = rtop*np.arccos(sinlatt**2 + (coslatt**2)*np.cos(d2r*(e - w)))
    Llat = rtop*np.arccos(np.sin(d2r*n)*np.sin(d2r*s) +
                          np.cos(d2r*n)*np.cos(d2r*s))
    Lr = top - bottom
    return distance, Llon, Llat, Lr


@numba.jit(nopython=True)
def split(w, e, s, n, top, bottom, nlon, nlat, nr, stack, stktop):
    """
    Divide the region into smaller parts and add them to the stack.
    """
    dlon = (e - w)/nlon
    dlat = (n - s)/nlat
    dr = (top - bottom)/nr
    for i in range(nlon):
        for j in range(nlat):
            for k in range(nr):
                stktop += 1
                stack[stktop, 0] = w + i*dlon
                stack[stktop, 1] = w + (i + 1)*dlon
                stack[stktop, 2] = s + j*dlat
                stack[stktop, 3] = s + (j + 1)*dlat
                stack[stktop, 4] = bottom + (k + 1)*dr
                stack[stktop, 5] = bottom + k*dr
    return stktop


@numba.jit(nopython=True)
def divisions(distance, Llon, Llat, Lr, ratio):
    "How many divisions should be made per dimension"
    nlon = 1
    nlat = 1
    nr = 1
    error = 0
    if distance <= ratio*Llon:
        if Llon <= 0.1:  # in meters. ~1e-6  degrees
            error = -1
        else:
            nlon = 2
    if distance <= ratio*Llat:
        if Llat <= 0.1:  # in meters. ~1e-6  degrees
            error = -1
        else:
            nlat = 2
    if distance <= ratio*Lr:
        if Lr <= 1e3:
            error = -1
        else:
            nr = 2
    return nlon, nlat, nr, nlon*nlat*nr, error


@numba.jit(nopython=True)
def kernelV(lon, coslat, sinlat, radius, lonc, sinlatc, coslatc, rc):
    r_sqr = radius**2
    result = 0
    for i in range(2):
        coslon = np.cos(lon - lonc[i])
        for j in range(2):
            cospsi = sinlat*sinlatc[j] + coslat*coslatc[j]*coslon
            for k in range(2):
                l_sqr = r_sqr + rc[k]**2 - 2*radius*rc[k]*cospsi
                kappa = (rc[k]**2)*coslatc[j]
                result += kappa/np.sqrt(l_sqr)
    return result


@numba.jit(nopython=True)
def kernelx(lon, coslat, sinlat, radius, lonc, sinlatc, coslatc, rc):
    r_sqr = radius**2
    result = 0
    for i in range(2):
        coslon = np.cos(lon - lonc[i])
        for j in range(2):
            kphi = coslat*sinlatc[j] - sinlat*coslatc[j]*coslon
            cospsi = sinlat*sinlatc[j] + coslat*coslatc[j]*coslon
            for k in range(2):
                l_sqr = r_sqr + rc[k]**2 - 2*radius*rc[k]*cospsi
                kappa = (rc[k]**2)*coslatc[j]
                result += kappa*rc[k]*kphi/(l_sqr**1.5)
    return result


@numba.jit(nopython=True)
def kernely(lon, coslat, sinlat, radius, lonc, sinlatc, coslatc, rc):
    r_sqr = radius**2
    result = 0
    for i in range(2):
        coslon = np.cos(lon - lonc[i])
        sinlon = np.sin(lonc[i] - lon)
        for j in range(2):
            cospsi = sinlat*sinlatc[j] + coslat*coslatc[j]*coslon
            for k in range(2):
                l_sqr = r_sqr + rc[k]**2 - 2*radius*rc[k]*cospsi
                kappa = (rc[k]**2)*coslatc[j]
                result += kappa*(rc[k]*coslatc[j]*sinlon/(l_sqr**1.5))
    return result


@numba.jit(nopython=True)
def kernelz(lon, coslat, sinlat, radius, lonc, sinlatc, coslatc, rc):
    r_sqr = radius**2
    result = 0
    for i in range(2):
        coslon = np.cos(lon - lonc[i])
        for j in range(2):
            cospsi = sinlat*sinlatc[j] + coslat*coslatc[j]*coslon
            for k in range(2):
                l_sqr = r_sqr + rc[k]**2 - 2*radius*rc[k]*cospsi
                kappa = (rc[k]**2)*coslatc[j]
                result += kappa*(rc[k]*cospsi - radius)/(l_sqr**1.5)
    # Multiply by -1 so that z is pointing down for gz and the gravity anomaly
    # doesn't look inverted (ie, negative for positive density)
    result *= -1
    return result


@numba.jit(nopython=True)
def kernelxx(lon, coslat, sinlat, radius, lonc, sinlatc, coslatc, rc):
    r_sqr = radius**2
    result = 0
    for i in range(2):
        coslon = np.cos(lon - lonc[i])
        for j in range(2):
            kphi = coslat*sinlatc[j] - sinlat*coslatc[j]*coslon
            cospsi = sinlat*sinlatc[j] + coslat*coslatc[j]*coslon
            for k in range(2):
                l_sqr = r_sqr + rc[k]**2 - 2*radius*rc[k]*cospsi
                kappa = (rc[k]**2)*coslatc[j]
                result += kappa*(3*((rc[k]*kphi)**2) - l_sqr)/(l_sqr**2.5)
    return result


@numba.jit(nopython=True)
def kernelxy(lon, coslat, sinlat, radius, lonc, sinlatc, coslatc, rc):
    r_sqr = radius**2
    result = 0
    for i in range(2):
        coslon = np.cos(lonc[i] - lon)
        sinlon = np.sin(lonc[i] - lon)
        for j in range(2):
            kphi = coslat*sinlatc[j] - sinlat*coslatc[j]*coslon
            cospsi = sinlat*sinlatc[j] + coslat*coslatc[j]*coslon
            for k in range(2):
                rc_sqr = rc[k]**2
                l_sqr = r_sqr + rc_sqr - 2*radius*rc[k]*cospsi
                kappa = rc_sqr*coslatc[j]
                result += kappa*3*rc_sqr*kphi*coslatc[j]*sinlon/(l_sqr**2.5)
    return result


@numba.jit(nopython=True)
def kernelxz(lon, coslat, sinlat, radius, lonc, sinlatc, coslatc, rc):
    r_sqr = radius**2
    result = 0
    for i in range(2):
        coslon = np.cos(lon - lonc[i])
        for j in range(2):
            kphi = coslat*sinlatc[j] - sinlat*coslatc[j]*coslon
            cospsi = sinlat*sinlatc[j] + coslat*coslatc[j]*coslon
            for k in range(2):
                rc_sqr = rc[k]**2
                l_5 = (r_sqr + rc_sqr - 2*radius*rc[k]*cospsi)**2.5
                kappa = rc_sqr*coslatc[j]
                result += kappa*3*rc[k]*kphi*(rc[k]*cospsi - radius)/l_5
    return result


@numba.jit(nopython=True)
def kernelyy(lon, coslat, sinlat, radius, lonc, sinlatc, coslatc, rc):
    r_sqr = radius**2
    result = 0
    for i in range(2):
        coslon = np.cos(lonc[i] - lon)
        sinlon = np.sin(lonc[i] - lon)
        for j in range(2):
            cospsi = sinlat*sinlatc[j] + coslat*coslatc[j]*coslon
            for k in range(2):
                rc_sqr = rc[k]**2
                l_sqr = r_sqr + rc_sqr - 2*radius*rc[k]*cospsi
                kappa = rc_sqr*coslatc[j]
                deltay = rc[k]*coslatc[j]*sinlon
                result += kappa*(3*(deltay**2) - l_sqr)/(l_sqr**2.5)
    return result


@numba.jit(nopython=True)
def kernelyz(lon, coslat, sinlat, radius, lonc, sinlatc, coslatc, rc):
    r_sqr = radius**2
    result = 0
    for i in range(2):
        coslon = np.cos(lonc[i] - lon)
        sinlon = np.sin(lonc[i] - lon)
        for j in range(2):
            cospsi = sinlat*sinlatc[j] + coslat*coslatc[j]*coslon
            for k in range(2):
                rc_sqr = rc[k]**2
                l_sqr = r_sqr + rc_sqr - 2*radius*rc[k]*cospsi
                kappa = rc_sqr*coslatc[j]
                deltay = rc[k]*coslatc[j]*sinlon
                deltaz = rc[k]*cospsi - radius
                result += kappa*3.*deltay*deltaz/(l_sqr**2.5)
    return result


@numba.jit(nopython=True)
def kernelzz(lon, coslat, sinlat, radius, lonc, sinlatc, coslatc, rc):
    r_sqr = radius**2
    result = 0
    for i in range(2):
        coslon = np.cos(lon - lonc[i])
        for j in range(2):
            cospsi = sinlat*sinlatc[j] + coslat*coslatc[j]*coslon
            for k in range(2):
                rc_sqr = rc[k]**2
                l_sqr = r_sqr + rc_sqr - 2*radius*rc[k]*cospsi
                l_5 = l_sqr**2.5
                kappa = rc_sqr*coslatc[j]
                deltaz = rc[k]*cospsi - radius
                result += kappa*(3*deltaz**2 - l_sqr)/l_5
    return result


# Use the factory to make the functions for specific fields. These are the ones
# that will be used by fatiando.gravmag.tesseroid
gx = engine_factory(kernelx)
gy = engine_factory(kernely)
gz = engine_factory(kernelz)
gxx = engine_factory(kernelxx)
gxy = engine_factory(kernelxy)
gxz = engine_factory(kernelxz)
gyy = engine_factory(kernelyy)
gyz = engine_factory(kernelyz)
gzz = engine_factory(kernelzz)
potential = engine_factory(kernelV)
