# cython: profile=True
# cython: linetrace=True
# distutils: define_macros=CYTHON_TRACE_NOGIL=0

from libc.math cimport floor
import cython
import numpy as np
cimport numpy as np
from cpython cimport array

from cspark_fof_c cimport in_buffer 
from spark_fof.fof cimport cfof
from spark_fof.fof.cfof cimport Particle
from bisect import bisect_left


DTYPE = np.float

ctypedef np.float_t DTYPE_f
ctypedef np.float32_t DTYPE_f32
ctypedef np.int_t DTYPE_i

pdt = np.dtype([('pos', 'f4', 3), ('is_ghost', 'i4'), ('iOrder', 'i8'), ('iGroup', 'i8')], align=True)

cdef unsigned int PRIMARY_GHOST_PARTICLE = 1
cdef unsigned int GHOST_PARTICLE_COPY = 2 

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef int get_bin_cython(float[3] point, int nbins, double[:] mins, double[:] maxs) nogil:
    cdef double dx, dy, dz, xbin, ybin, zbin
    cdef bint in_bounds = 1
    
    # check bounds 
    for i in range(3): 
        in_bounds *= mins[i] <= point[i] <= maxs[i]
    if not in_bounds: 
        return -1
    
    dx = (maxs[0] - mins[0]) / <double>nbins
    dy = (maxs[1] - mins[1]) / <double>nbins
    dz = (maxs[2] - mins[2]) / <double>nbins
    xbin = floor((point[0] - mins[0]) / dx)
    ybin = floor((point[1] - mins[1]) / dy)
    zbin = floor((point[2] - mins[2]) / dz)
    return <int>(xbin + ybin * nbins + zbin*nbins*nbins)

cpdef int get_bin_wrapper(float[:] point, int nbins, double[:] mins, double[:] maxs):
    cdef float[3] point2
    cdef unsigned int i
    for i in range(3): 
        point2[i] = point[i]

    return get_bin_cython(point2, nbins, mins, maxs)

@cython.boundscheck(False)
cdef bint rect_buffer_zone_cython(float[3] point, 
                                  double* mins, double* maxs, 
                                  double* mins_buff, double* maxs_buff) nogil:
    """Determine whether a particle is in the buffer zone"""
    cdef bint in_main, in_buffer

    in_main = in_rectangle_cython(mins, maxs, point)
    in_buffer = in_rectangle_cython(mins_buff, maxs_buff, point)
    return (in_main != in_buffer)


@cython.profile(False)
@cython.boundscheck(False)
cdef inline bint in_rectangle_cython(double[3] mins, double[3] maxs, float[3] point) nogil:
    """Check if particle is inside a rectangle"""
    cdef float size
    cdef bint res=1
    cdef unsigned int i, ndim=3

    for i in range(ndim): 
        res *= (mins[i] <= point[i] < maxs[i])
    return res


@cython.boundscheck(False)
cpdef remap_gid_partition_cython(Particle [:] p_arr, dict gid_map):
    """Remap group ids"""
    cdef long g
    cdef int i

    for i in range(p_arr.shape[0]):
        g = p_arr[i].iGroup
        if g in gid_map:
            p_arr[i].iGroup = gid_map[g]


@cython.boundscheck(False)
def count_groups_cython(Particle [:] p_arr):
    """Count particles per group and yield one group at a time"""
    cdef int i
    cdef long [:] gc_mv
    cdef long [:] counts_mv

    gc, counts = np.unique(np.asarray(p_arr)['iGroup'], return_counts=True)

    gc_mv = gc
    counts_mv = counts

    for i in range(gc_mv.shape[0]):
        yield (gc_mv[i], counts_mv[i])


@cython.boundscheck(False)
def count_groups_partition_cython(particle_arrays, gr_map_inv_b, nMinMembers): 
    gids, counts = np.unique(np.concatenate(list(particle_arrays))['iGroup'], return_counts=True)
    return ((g,cnt) for (g,cnt) in zip(gids,counts) if (g in gr_map_inv_b.value) or (cnt >= nMinMembers))


@cython.boundscheck(False)
cdef long count_ghosts(Particle [:] p_arr) nogil: 
    """Return the number of ghost particles in the array"""
    cdef long nghosts=0
    cdef unsigned int i
    for i in range(p_arr.shape[0]): 
        if p_arr[i].is_ghost: nghosts+=1
    return nghosts


@cython.boundscheck(False)
def partition_array(Particle[:] p_arr, int N, float tau, int symmetric,
                    double[:] dom_mins, 
                    double[:] dom_maxs):
    """
    Split a particle array into tuples of (group_id, array).
    This can be used to repartition the data in order to get all particles
    for a given group ID on the same partition. 
    """
    
    cdef unsigned int i 
    cdef long right_ind, left_ind
    
    with nogil:
        for i in range(p_arr.shape[0]):
            p_arr[i].iGroup = get_bin_cython(p_arr[i].r, N, dom_mins, dom_maxs)
            
    p_np = np.array(p_arr)
    p_np.sort(order='iGroup')
    partitions = np.unique(p_np['iGroup'])

    left_ind = 0
    res = []
    for partition_ind in range(len(partitions)):
        if partition_ind == len(partitions)-1: 
            res.append((partitions[partition_ind], p_np[left_ind:]))
        else : 
            right_ind = bisect_left(p_np['iGroup'], partitions[partition_ind+1])
            res.append((partitions[partition_ind], p_np[left_ind:right_ind]))
            left_ind=right_ind
    return res


@cython.boundscheck(False)
def partition_ghosts(Particle[:] p_arr, int N, float tau, int symmetric,
                            double[:] dom_mins, 
                            double[:] dom_maxs):
    cdef unsigned int i, j, k, ghost_ind=0, n, trans_max
    cdef long right_ind, left_ind
    cdef int my_bin, trans_bin
    cdef float* point
    cdef float[:, :] trans = np.array([[-tau, 0, 0], 
                                       [0,-tau, 0], 
                                       [0, 0, -tau], 
                                       [-tau, -tau, 0], 
                                       [0, -tau, -tau], 
                                       [-tau,-tau,-tau],
                                       [tau, 0, 0], 
                                       [0,tau, 0], 
                                       [0, 0, tau], 
                                       [tau, tau, 0], 
                                       [0, tau, tau], 
                                       [tau,tau,tau]], 
                                  dtype=np.float32)
    cdef float[3] new_point
    cdef Particle[:] ghosts 
    cdef Particle ghost_particle
    cdef int[:] trans_mark = np.zeros(trans.shape[0], dtype=np.int32)
    
    # check the number of ghost particles we have
    nghosts = count_ghosts(p_arr)
    ghosts = np.zeros(nghosts*4, dtype=pdt)
    
    trans_max = 12 if symmetric else 6

    with nogil:
        for i in range(p_arr.shape[0]):
            point = p_arr[i].r
            my_bin = p_arr[i].iGroup # the bin ID is stored in the iGroup field at read time
            
            if p_arr[i].is_ghost:
                ghost_particle = p_arr[i]

                trans_mark[:] = -1

                for j in range(trans_max):
                    for k in range(3):
                        new_point[k] = p_arr[i].r[k] + trans[j,k]
                    
                    trans_bin = get_bin_cython(new_point, N, dom_mins, dom_maxs)

                    if trans_bin != my_bin and \
                       trans_bin >= 0 \
                       and not bin_already_there(trans_mark, trans_bin, trans.shape[0]): 

                        trans_mark[j] = trans_bin
                        ghost_particle.iGroup = trans_bin
                        ghost_particle.is_ghost = GHOST_PARTICLE_COPY
                        ghosts[ghost_ind] = ghost_particle
                        ghost_ind+=1
                                                          
    all_ps = np.asarray(ghosts)[np.nonzero(ghosts)[0]]

    all_ps.sort(order='iGroup')
    partitions = np.unique(all_ps['iGroup'])
   # print 'partitions: ', partitions

    left_ind = 0
    res = []
    for partition_ind in range(len(partitions)):
        if partition_ind == len(partitions)-1: 
            res.append((partitions[partition_ind], all_ps[left_ind:]))
        else : 
            right_ind = bisect_left(all_ps['iGroup'], partitions[partition_ind+1])
            res.append((partitions[partition_ind], all_ps[left_ind:right_ind]))
            left_ind=right_ind
    return res


@cython.profile(False)
@cython.boundscheck(False)
cdef inline bint bin_already_there(int[:] trans_mark, int bin, unsigned int ndim) nogil: 
    cdef unsigned int i
    
    for i in range(ndim): 
        if trans_mark[i] == bin: return 1
    return 0

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef void ghost_mask(Particle[:] p_arr, float tau, int N,
               double[:, :] mins,
               double[:, :] maxs,
               double[:, :] mins_buff,
               double[:, :] maxs_buff, 
               double[:] dom_mins, 
               double[:] dom_maxs):

    cdef unsigned int n_containers = mins.shape[0]
    cdef unsigned int nparts, i
    cdef int my_bin
    cdef float* point
    
    for i in range(p_arr.shape[0]):
        point = p_arr[i].r
        my_bin = get_bin_cython(point, N, dom_mins, dom_maxs)
        p_arr[i].is_ghost = rect_buffer_zone_cython(point, 
                                                    &mins[my_bin,0], 
                                                    &maxs[my_bin,0], 
                                                    &mins_buff[my_bin,0], 
                                                    &maxs_buff[my_bin,0])


@cython.boundscheck(False)
cpdef void encode_gid(Particle [:] p_arr, unsigned long partition_index) nogil:
    cdef unsigned int i
    cdef long gid
    cdef Particle p
    
    for i in range(p_arr.shape[0]):
        gid = p_arr[i].iGroup
        p_arr[i].iGroup = partition_index<<32 | gid


@cython.boundscheck(False)
def relabel_groups(Particle [:] p_arr, groups_map): 
    cdef long i
    cdef long g
    
    for i in range(p_arr.shape[0]):
        g = p_arr[i].iGroup
        if g in groups_map:
            p_arr[i].iGroup = groups_map[g]
        else: 
            p_arr[i].iGroup = 0
            