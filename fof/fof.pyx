cimport cfof

cdef kdInit(cfof.KD* kd, int nBucket, float* fPeriod):
	print nBucket, fPeriod[0]
	cfof.kdInit(kd, nBucket, fPeriod)

cpdef initialize():
	cdef cfof.KD kd
	cdef float fPeriod[3]
	cdef int nBucket = 16

	fPeriod[0] = 1
	fPeriod[1] = 1
	fPeriod[2] = 1

	kdInit(&kd, nBucket, fPeriod)