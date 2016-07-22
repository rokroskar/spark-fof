# cython: profile=True 

from libc.math cimport floor
import cython
import numpy as np
cimport numpy as np

DTYPE = np.float

ctypedef np.float_t DTYPE_f
ctypedef np.int_t DTYPE_i

pdt = np.dtype([('pos', 'f4', 3), ('iGroup', 'i8'), ('iOrder', 'i4')], align=True)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef get_bin_cython(np.ndarray[float] point, int nbins, np.ndarray[DTYPE_f] mins, np.ndarray[DTYPE_f] maxs):
    cdef float dx, dy, dz, xbin, ybin, zbin
    cdef bint in_bounds = 1
    cdef int ndim = point.shape[0]
    # check bounds 
    for i in range(3): 
        in_bounds *= mins[i] <= point[i] <= maxs[i]
    if not in_bounds: 
        return -1
    # if not (px >= minx and px <= maxx and py >= miny and py <= maxy and pz >= minz and pz <= maxz):
    #     return -1

    dx = (maxs[0] - mins[0]) / <float>nbins
    dy = (maxs[1] - mins[1]) / <float>nbins
    dz = (maxs[2] - mins[2]) / <float>nbins
    xbin = floor((point[0] - mins[0]) / dx)
    ybin = floor((point[1] - mins[1]) / dy)
    zbin = floor((point[2] - mins[2]) / dz)
    return <int>(xbin + ybin * nbins + zbin*nbins*nbins)
    

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def get_particle_bins_cython(np.ndarray[float, ndim=3] points, 
                             np.ndarray[DTYPE_i] bins,
                             np.ndarray[DTYPE_f] mins,
                             np.ndarray[DTYPE_f] maxs): 
    cdef int n = len(points)
    for i in range(n): 
        bins[i] = get_bin_cython(points[i], 100, mins, maxs)


def rect_buffer_zone_cython(np.ndarray[float] point, rect):
    """Determine whether a particle is in the buffer zone"""
    cdef bint in_main, in_buffer
    cdef np.ndarray[DTYPE_f] mins, maxs, mins_buff, maxs_buff
    
    mins = rect.mins
    maxs = rect.maxes
    mins_buff = rect.bufferRectangle.mins
    maxs_buff = rect.bufferRectangle.maxes
    
    in_main = in_rectangle_cython(mins, maxs, point)
    in_buffer = in_rectangle_cython(mins_buff, maxs_buff, point)
    return (in_main != in_buffer)


def partition_particles_cython(particles, domain_containers, tau, dom_mins, dom_maxs):
    """Copy particles in buffer areas to the partitions that will need them"""
    cdef int N = domain_containers[0].N
    cdef int nparts, i

    p_arr = np.fromiter(particles, pdt)
    nparts = len(p_arr)
    
    trans = np.array([[-tau, 0, 0], [0,-tau, 0], [0, 0, -tau], [-tau, -tau, 0], [0, -tau, -tau], [-tau,-tau,-tau]], dtype=np.float32)

    for i in range(nparts):
        point = p_arr['pos'][i]
        my_bins = []
        my_bins.append(get_bin_cython(point, 2**N, dom_mins, dom_maxs))

        my_rect = domain_containers[my_bins[0]]

        if rect_buffer_zone_cython(point, my_rect):
            # particle coordinates in single array
            coords = np.copy(point)
            # iterate through the transformations
            for t in trans: 
                #x,y,z = coords + t
                trans_bin = get_bin_cython(coords+t, 2**N, dom_mins,dom_maxs)
                if trans_bin not in my_bins and trans_bin >= 0:
                    my_bins.append(trans_bin)
                    yield (trans_bin, p_arr[i])

        # return the first bin, i.e. the only non-ghost bin
        yield (my_bins[0], p_arr[i])


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef in_rectangle_cython(np.ndarray[DTYPE_f] mins, np.ndarray[DTYPE_f] maxs, np.ndarray[float] point):
    cdef float size
    cdef bint res=1
    cdef int i

    for i in range(3): 
        res *= mins[i] < point[i] < maxs[i]
    return res

def remap_gid_partition_cython(particles, gid_map):
    cdef np.int64_t g
    
    p_arr = np.fromiter(particles, pdt)
    
    for i in range(len(p_arr)):
        g = p_arr['iGroup'][i]
        if g in gid_map:
            p_arr['iGroup'][i] = gid_map[g]
    return p_arr

