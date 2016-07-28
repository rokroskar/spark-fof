cdef inline bint in_buffer(double* mins, float* point, float tau): 
    return (point[0] <= mins[0] + tau) | \
           (point[1] <= mins[1] + tau) | \
           (point[2] <= mins[2] + tau)
