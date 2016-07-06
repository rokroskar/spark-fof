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
		int nLevels
		int nNodes
		int nSplit
		PARTICLE *p
		KDN *kdNodes
		int nGroup
		int uSecond
		int uMicro
		
	ctypedef kdContext* KD 

	int kdInit(KD*, int, float*)

	int foo(int)