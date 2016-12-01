"""
Classes for domain handling in spark-fof
"""
from scipy.spatial import Rectangle
import numpy as np

def setup_domain(N, tau, mins, maxes):
    """Set up the rectangles that define the domain"""

    domain_containers = []

    n_containers = N**3

    xbins = np.linspace(mins[0], maxes[0], N+1)
    ybins = np.linspace(mins[1], maxes[1], N+1)
    zbins = np.linspace(mins[2], maxes[2], N+1)
    
    for i in range(N): 
        for j in range(N):
            for k in range(N): 
                domain_containers.append(DomainRectangle([xbins[k], ybins[j], zbins[i]],
                                                         [xbins[k+1],   ybins[j+1],   zbins[i+1]], tau=tau, N=N))

    return domain_containers


class DomainRectangle(Rectangle):
    def __init__(self, mins, maxes, N=None, parent=None, tau=0.1, symmetric=False):
        self.parent = parent
        super(DomainRectangle, self).__init__(mins, maxes)
        self.children = []
        self.midpoint = self.mins + (self.maxes - self.mins) / 2.

        if N is None:
            self.N = 0
        else:
            self.N = N

        self.tau = tau

        if symmetric:
            self.bufferRectangle = Rectangle(self.mins + tau, self.maxes - tau)
        else: 
            self.bufferRectangle = Rectangle(self.mins + tau, self.maxes)

    def __repr__(self):
        return "<DomainRectangle %s>" % list(zip(self.mins, self.maxes))

    def get_inner_box(self, tau):
        """Return a new hyper rectangle, shrunk by tau"""

        new_rect = copy.copy(self)
        new_rect.mins = self.mins + tau
        new_rect.maxes = self.maxes - tau

        return new_rect

    def split(self, d, split, N):
        """
        Produce two hyperrectangles by splitting.
        In general, if you need to compute maximum and minimum
        distances to the children, it can be done more efficiently
        by updating the maximum and minimum distances to the parent.
        Parameters
        ----------
        d : int
            Axis to split hyperrectangle along.
        split : float
            Position along axis `d` to split at.
        """
        mid = np.copy(self.maxes)
        mid[d] = split
        less = DomainRectangle(self.mins, mid, N=N, tau=self.tau)
        mid = np.copy(self.mins)
        mid[d] = split
        greater = DomainRectangle(mid, self.maxes, N=N, tau=self.tau)

        return less, greater

    def split_domain(self, max_N=1, N=1):
        ndim = len(self.maxes)

        # Keep splitting until max level is reached
        if N <= max_N:
            split_point = self.mins + (self.maxes - self.mins) / 2
            rs = self.split(0, split_point[0], N)

            # split along all dimensions
            for axis in range(1, ndim):
                rs = [r.split(axis, split_point[axis], N) for r in rs]

                if isinstance(rs[0], (tuple, list)):
                    rs = [item for sublist in rs for item in sublist]

            self.children = rs

            for r in rs:
                r.parent = self

            res = flatten([r.split_domain(max_N, N + 1) for r in rs])

            return res

        else:
            return self

def flatten(S):
    if S == []:
        return S
    if isinstance(S[0], list):
        return flatten(S[0]) + flatten(S[1:])
    return S[:1] + flatten(S[1:])


    # def in_buffer_zone(self, p):
    #     """Determine whether a particle is in the buffer zone"""
    #     return rect_buffer_zone_cython(p['pos'], self.mins, self.maxes, self.bufferRectangle.mins, self.bufferRectangle.maxes)
