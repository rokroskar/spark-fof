from scipy.spatial import Rectangle
import numpy as np
from math import floor, ceil
from functools import total_ordering
import networkx as nx

# encode a 32-bit partition ID (pid) and 32-bit cluster ID (cid) into one 64-bit integer
def encode_gid(pid, cid, bits=32):
    if bits == 32: 
        res = np.int64(int(np.binary_repr(pid,width=32)+np.binary_repr(cid,width=32),2))
    elif bits == 16:
        res = np.int32(int(np.binary_repr(pid,width=16)+np.binary_repr(cid,width=16),2))
    else: 
        raise RuntimeError('Group encoding must use either 16 or 32 bit integers')
    return res

# decode a two-part 64-bit integer into a 32-bit partition ID (pid) and 32-bit cluster ID (cid)
def decode_gid(gid, bits = 32):
    if (bits is not 32) and (bits is not 16): 
        raise RuntimeError('Group encoding must use either 16 or 32 bit integers')
    pid = gid >> bits
    cid = gid - (pid << bits)
    return pid,cid


# define wrapped cython functions
#get_bin_cython = spark_cython('spark_fof_c', 'get_bin_cython')
#get_particle_bins_cython = spark_cython('spark_fof_c', 'get_particle_bins_cython')
#rect_buffer_zone_cython = spark_cython('spark_fof_c', 'rect_buffer_zone_cython')
#partition_particles_cython = spark_cython('spark_fof_c', 'partition_particles_cython')
#run_fof = spark_cython('fof', 'run')

from spark_fof_c import get_bin_cython, \
                        get_particle_bins_cython, \
                        rect_buffer_zone_cython, \
                        partition_particles_cython, \
                        remap_gid_partition_cython, \
                        pdt

import fof

class FOFAnalyzer():
    def __init__(self, sc, N, tau, particle_rdd, mins, maxs):
        self.sc = sc
        self.N = N
        self.tau = tau
        self.domain_containers = setup_domain(N, tau, maxs, mins)

        self.particle_rdd = particle_rdd

    def run_local_fof(self):
        pass

    def get_gid_map(self, level=0):
        """
        Take a particle RDD and return a gid -> gid' that will link groups in the buffer region.

        This is done in two steps: 
            - first, a mapping of buffer region particle IDs to group IDs is made
            - second, the particle IDs corresponsing to the buffer region particles
              are filtered from the full data and a map is produced that maps all groups
              onto the group corresponding to the lowest container ID

        Inputs:

        particle_rdd: an RDD of particles

        domain_containers: sorted list of domain hyper rectangles

        Returns:

        list of tuples of (src,dst) group ID mappings 
        """
        particle_rdd = self.particle_rdd
        domain_containers = self.domain_containers
        sc = self.sc

        pg_map = (particle_rdd.mapPartitionsWithIndex(
                  lambda index, data: get_buffer_particles(index,
                                                           data,
                                                           domain_containers, level))
                  .map(pid_gid)
                  .collectAsMap())

        pg_map_b = sc.broadcast(set(pg_map.keys()))

        # generate the "local" groups mapping -- this will only link groups among neighboring domains
        # this proceeds in a few stages:
        #
        # 1. filter only the ghost particles and return a (pid, gid) key,value pair RDD
        # 2. for each ghost particle pid, aggregate together all of its groups
        # 3. from each group list, generate a (g, g') key, value pair RDD where
        # g maps onto g'

        N_partitions = sc.defaultParallelism*20

        groups_map = (particle_rdd.map(pid_gid)
                                  .filter(lambda (pid, gid): pid in pg_map_b.value)
                                  .aggregateByKey([], lambda l, g: l + [g], lambda a, b: sorted(a + b))
                                  .values()
                                  .flatMap(lambda gs: [(g, gs[0]) for g in gs[1:]])).collect()

        return groups_map

 
    def get_level_map(self, level=0):
        """Produce a group re-mapping across sub-domains. Connected groups are obtained by finding
        groups belonging to the same particles and linking them into a graph. Each node in a 
        connected sub-graph is mapped to the lowest group ID in the sub-graph. 

        Inputs: 

        particle_rdd: of particles

        domain_containers: sorted list of domain hyper rectangles

        Optional Keywords: 

        level: how many levels up from the base domain to begin; starting at 
               a higher level might reduce the total number of groups that 
               need to be considered. Default is 0, meaning that the merging 
               takes place at the finest sub-domain level. 
        """
        particle_rdd = self.particle_rdd
        domain_containers = self.domain_containers

        # get the initial group mapping across sub-domains just based on
        # particle IDs
        groups_map = self.get_gid_map(level)

        mappings = {}

        if len(groups_map) > 0:
            src, dst = zip(*groups_map)

            # generate the graph
            g = nx.Graph()
            g.add_nodes_from(src + dst)

            for e in groups_map:
                g.add_edge(*e)

            # process the connected components
            for sg in nx.connected.connected_component_subgraphs(g):
                if len(sg) > 1:
                    # generate mapping to lowest-common-group
                    base_node = min(sg.nodes())
                    new_mapping = {
                        node: base_node for node in sg.nodes() if node != base_node}
                    mappings.update(new_mapping)

        return mappings

    def merge_groups(self, level=0):
        """
        For an RDD of particles, discover the groups connected across domain
        boundaries and remap to a lowest common group ID. 

        Inputs: 

        particle_rdd: RDD of Particles

        domain_containers: sorted list of domain hyper rectangles

        Optional Keywords:

        level: how many levels up from the base domain to begin; starting at 
               a higher level might reduce the total number of groups that 
               need to be considered. Default is 0, meaning that the merging 
               takes place at the finest sub-domain level. 
        """
        particle_rdd = self.particle_rdd
        domain_containers = self.domain_containers

        for l in range(level, -1, -1):
            m = self.get_level_map(l)
            m_b = self.sc.broadcast(m)
            particle_rdd = particle_rdd.mapPartitions(lambda particles: remap_gid_partition_cython(particles, m_b.value))

        return particle_rdd


def setup_domain(N, tau, maxes, mins):
    D = DomainRectangle(maxes, mins, tau=tau)
    domain_containers = D.split_domain(max_N=N)
    for r in domain_containers:
        r.bin = get_rectangle_bin(r, D.mins, D.maxes, 2**N)

    domain_containers.sort(key=lambda x: x.bin)

    return domain_containers


def partition_particles(particles, domain_containers, tau, mins, maxs):
    """Copy particles in buffer areas to the partitions that will need them"""

    N = domain_containers[0].N

    trans = np.array([[-tau, 0, 0], [0,-tau, 0], [0, 0, -tau], [-tau, -tau, 0], [0, -tau, -tau], [-tau,-tau,-tau]])

    for p in particles:
        pos = p['pos']
        my_bins = []
        my_bins.append(get_bin_cython(pos, 2**N, mins, maxs))

        my_rect = domain_containers[my_bins[0]]

        if rect_buffer_zone_cython(pos,domain_containers):
            # particle coordinates in single array
           # coords = np.copy(pos)
            # iterate through the transformations
            for t in trans: 
#                x,y,z = coords + t
                trans_bin = get_bin_cython(pos+t, 2**N, mins, maxs)
                if trans_bin not in my_bins and trans_bin > 0:
                    my_bins.append(trans_bin)
                    yield (trans_bin, p)

        # return the first bin, i.e. the only non-ghost bin
        yield (my_bins[0], p)


def get_bin(px, py, pz, nbins, mins, maxs):
    minx, miny, minz = mins
    maxx, maxy, maxz = maxs

    if not all([minx <= px <= maxx, miny <= py <= maxy, minz <= pz <= maxz]):
        return -1

    dx = (maxx - minx) / float(nbins)
    dy = (maxy - miny) / float(nbins)
    dz = (maxz - minz) / float(nbins)
    xbin = floor((px - minx) / dx)
    ybin = floor((py - miny) / dy)
    zbin = floor((pz - minz) / dz)
    return int(xbin + ybin*nbins + zbin*nbins*nbins)


def flatten(S):
    if S == []:
        return S
    if isinstance(S[0], list):
        return flatten(S[0]) + flatten(S[1:])
    return S[:1] + flatten(S[1:])


def get_rectangle_bin(rec, mins, maxs, nbins):
    # take the midpoint of the rectangle
    point = rec.mins + (rec.maxes - rec.mins) / 2.
    return get_bin_cython(point.astype(np.float32), nbins, mins, maxs)


def get_buffer_particles(partition, particles, domain_containers, level=0):
    """Produce the particles from the buffer regions"""
    my_rect = domain_containers[partition]

    # get to the correct level
    for i in range(level):
        my_rect = my_rect.parent

    for p in particles:
        if my_rect.in_buffer_zone(p):
            yield p


def pid_gid(p):
    """Map the particle to its pid and gid"""
    return (p['iOrder'], p['iGroup'])


def remap_gid(p, gid_map):
    """Remap gid if it exists in the map"""
    if p['iGroup'] in gid_map.keys():
        p_c = np.copy(p)
        p_c['iGroup'] = gid_map[p['iGroup']]
        return p_c
    else: 
        return p

def set_local_group(partition, particles):
    """Set an initial partition for the group ID"""
    p_arr = np.fromiter(particles, pdt)
    p_arr['gid'] = encode_gid(partition, 0)
    return iter(p_arr)


#################
#
# DOMAIN CLASSES
#
#################


class DomainRectangle(Rectangle):
    def __init__(self, mins, maxes, N=None, parent=None, tau=0.1):
        self.parent = parent
        super(DomainRectangle, self).__init__(mins, maxes)
        self.children = []
        self.midpoint = self.mins + (self.maxes - self.mins) / 2.

        if N is None:
            self.N = 0
        else:
            self.N = N

        self.tau = tau

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

    def in_buffer_zone(self, p):
        """Determine whether a particle is in the buffer zone"""
        return rect_buffer_zone_cython(p['pos'], self)
