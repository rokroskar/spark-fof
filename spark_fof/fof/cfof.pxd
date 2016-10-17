from libcpp cimport bool

#
# types
#
cdef extern from "kd.h":
    cdef struct Particle:
        float r[3]
        int is_ghost
        int iOrder
        long iGroup
    ctypedef Particle PARTICLE


    cdef struct kdNode:
        pass
    ctypedef kdNode KDN


    cdef struct kdContext:
        int nBucket
        int nParticles
        int nDark
        int nGas
        int nStar
        int bDark
        int bGas
        int bStar
        int nActive
        float fTime;
        float fPeriod[3]
        float fCenter[3]
        int nLevels
        int nNodes
        int nSplit
        PARTICLE *p
        KDN *kdNodes
        int nGroup
        int uSecond
        int uMicro
    ctypedef kdContext* KD 

#
# functions
#
    int kdInit(KD*, int, float*, float*)
    void kdBuildTree(KD)
    void kdTime(KD, int*, int*)
    int kdFoF(KD, float)
    int kdTooSmall(KD,int)
    void kdOrder(KD)
    void kdFinish(KD)