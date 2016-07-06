from libc.math cimport floor
import cython
import numpy as np
cimport numpy as np

DTYPE = np.float

ctypedef np.float_t DTYPE_f
ctypedef np.int_t DTYPE_i

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def get_bin_cython(float px, float py, float pz, int nbins, float minx, 
                   float miny, float minz, float maxx, float maxy, float maxz):
    #if not all([minx <= px <= maxx, miny <= py <= maxy]):
    cdef float dx, dy, dz, xbin, ybin, zbin
    if not (px >= minx and px <= maxx and py >= miny and py <= maxy and pz >= minz and pz <= maxz):
        return -1

    dx = (maxx - minx) / <float>nbins
    dy = (maxy - miny) / <float>nbins
    dz = (maxz - minz) / <float>nbins
    xbin = floor((px - minx) / dx)
    ybin = floor((py - miny) / dy)
    zbin = floor((pz - minz) / dz)
    return <int>(xbin + ybin * nbins + zbin*nbins*nbins)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def get_particle_bins_cython(np.ndarray[DTYPE_f] xs, np.ndarray[DTYPE_f] ys, np.ndarray[DTYPE_i] bins): 
    cdef int n = xs.shape[0]
    for i in range(n): 
        bins[i] = get_bin_cython(xs[i], ys[i], 100, -1,-1,1,1)

