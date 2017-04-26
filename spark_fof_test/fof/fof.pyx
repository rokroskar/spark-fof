cimport cfof

# type declarations
import numpy as np
cimport numpy as np

DTYPE = np.float

ctypedef np.float_t DTYPE_f
ctypedef np.int_t DTYPE_i

cdef extern from "math.h":
    float INFINITY

# fof wrapper functions
cdef kdInit(cfof.KD* kd, int nBucket, float* fPeriod, float* fCenter):
    cfof.kdInit(kd, nBucket, fPeriod, fCenter)

cdef kdBuildTree(cfof.KD kd): 
    cfof.kdBuildTree(kd)

cdef kdTime(cfof.KD kd, int* sec, int* usec):
    cfof.kdTime(kd, sec, usec)

cdef kdFoF(cfof.KD kd, float fEps):
    return cfof.kdFoF(kd, fEps)

cdef kdTooSmall(cfof.KD kd, int nMembers):
    return cfof.kdTooSmall(kd, nMembers)

cdef kdOrder(cfof.KD kd):
    cfof.kdOrder(kd)

cdef kdFinish(cfof.KD kd):
    cfof.kdFinish(kd)

#
# cython functions for working with kd/fof
#

cdef populate_arrays(cfof.KD kd, np.ndarray[cfof.PARTICLE] particles):
    cdef cfof.PARTICLE[:] parr = particles

    kd.p = &parr[0]

    kd.nParticles = <int>len(particles)
    kd.nDark = <int>len(particles)
    kd.nGas = 0
    kd.nStar = 0
    kd.fTime = 0.0
    kd.nActive = kd.nDark
    kd.bDark = 1
    kd.bGas = 0
    kd.bStar = 0


# main function called from python
cpdef int run(np.ndarray[cfof.PARTICLE] particles, float fEps, int nMembers):
    cdef cfof.KD kd
    cdef float fPeriod[3] 
    cdef float fCenter[3]
    cdef int nBucket = 16
    cdef int res
    cdef int nGroup 
    cdef int sec, usec
    cdef int i

    for i in range(3): 
        fPeriod[i] = INFINITY
        fCenter[i] = 0.0

    # initialize
    kdInit(&kd, nBucket, fPeriod, fCenter)

    # put the arrays into the kd context
    populate_arrays(kd, particles)

    # build the tree
    kdBuildTree(kd)

    # set the timer
    kdTime(kd,&sec,&usec)

    # get groups
    nGroup = kdFoF(kd, fEps)

    # set the timer
    kdTime(kd,&sec,&usec)

    # eliminate small groups
    if nMembers > 1:
        nGroup = kdTooSmall(kd, nMembers)    

    return nGroup
