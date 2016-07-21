cdef inline bint within_limits(float* mins, float* maxs, float* point): 
    cdef bint inside = 1
    for i in range(3): 
        inside *= mins[i] < point[i] < maxs[i]
    return inside

cdef extern from "kd.h":
    cdef struct Particle:
        float r[3]
        int iGroup
        int iOrder

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

    int kdInit(KD*, int, float*, float*)
    void kdBuildTree(KD)
    void kdTime(KD, int*, int*)
    int kdFoF(KD, float)
    int kdTooSmall(KD,int)
    void kdOrder(KD)
    void kdFinish(KD)