from scipy.spatial import Rectangle
import numpy as np
from math import floor, ceil
from functools import total_ordering
import networkx as nx
from collections import defaultdict

# initialize spark to load spark classes
import findspark
findspark.init()
import pyspark

from . import spark_tipsy

import spark_fof_c

from spark_fof_c import  remap_gid_partition_cython, \
                         relabel_groups, \
                         ghost_mask, \
                         pdt
                         
from . import fof

PRIMARY_GHOST_PARTICLE = 1
GHOST_PARTICLE_COPY = 2 

class FOFAnalyzer():
    def __init__(self, sc, particles, nMinMembers, nBins, tau, mins=[-.5,-.5,-.5], maxs=[.5,.5,.5]):
        self.sc = sc
        self.nBins = nBins
        self.tau = tau
        self.mins = mins
        self.maxs = maxs
        self.nMinMembers = nMinMembers

        domain_containers = setup_domain(nBins, tau, maxs, mins)

        self.domain_containers = domain_containers

        if isinstance(particles, str): 
            # we assume we have a file location
            p_rdd = spark_tipsy.read_tipsy_output(sc, particles, chunksize=1024*4)
            self.particle_rdd = p_rdd
        
        elif isinstance(particles, pyspark.rdd.RDD):
            self.particle_rdd = particles

        # set up RDD place-holders
        self._partitioned_rdd = None
        self._fof_rdd = None
        self._merged_rdd = None
        self._final_fof_rdd = None
        self._groups = None

    # define RDD properties 
    @property
    def partitioned_rdd(self):
        if self._partitioned_rdd is None: 
            self._partitioned_rdd = self.partition_particles()
        return self._partitioned_rdd

    @property
    def fof_rdd(self):
        if self._fof_rdd is None:
            self._fof_rdd = self.run_fof()
        return self._fof_rdd

    @property
    def merged_rdd(self):
        if self._merged_rdd is None:
            self._merged_rdd = self.merge_groups()
        return self._merged_rdd
    

    @property
    def final_fof_rdd(self):
        if self._final_fof_rdd is None: 
            self._final_fof_rdd = self.finalize_groups()
        return self._final_fof_rdd

    
    @property
    def groups(self):
        if self._groups is None: 
            self.final_fof_rdd
        return self._groups

    
    
    def run_all(self): 
        """
        Run FOF, merge the groups across domains and finalize the group IDs,
        dropping ones that don't fit criteria.
        """



    def partition_particles(self): 
        """Partitions the particles for running local FOF"""

        Npartitions = len(self.domain_containers)
        tau, domain_containers, dom_mins, dom_maxs = self.tau, self.domain_containers, self.mins, self.maxs

        # set up domain limit arrays
        N = domain_containers[0].N
        n_containers = len(domain_containers)
        mins = np.zeros((n_containers, 3))
        maxs = np.zeros((n_containers, 3))
        mins_buff = np.zeros((n_containers, 3))
        maxs_buff = np.zeros((n_containers, 3))

        for i in range(n_containers): 
            mins[i] = domain_containers[i].mins
            maxs[i] = domain_containers[i].maxes
            mins_buff[i] = domain_containers[i].bufferRectangle.mins
            maxs_buff[i] = domain_containers[i].bufferRectangle.maxes

        # set up a helper function for calling the cython code
        def partition_wrapper(particle_iterator): 
            for particle_array in particle_iterator: 
                # first mark the ghosts
                ghost_mask(particle_array, tau, N, mins, maxs, mins_buff, maxs_buff, dom_mins, dom_maxs)
                res = spark_fof_c.new_partitioning_cython(particle_array, domain_containers, tau, dom_mins, dom_maxs)
                for r in res: 
                    yield r

        partitioned_rdd = (self.particle_rdd.mapPartitions(partition_wrapper)).partitionBy(Npartitions).values()

        self._partitioned_rdd = partitioned_rdd

        return partitioned_rdd

    def run_fof(self):
        """
        Run FOF on the particles 

        First does a partitioning step to put particles in their respective domain containers
        """
        tau = self.tau

        def run_local_fof(partition_index, particle_iter, tau, nMinMembers, batch_size=1024*256): 
            """Helper function to run FOF locally on the individual partitions"""
            part_arr = np.hstack(particle_iter)
            if len(part_arr)>0:
                # run fof
                fof.run(part_arr, tau, nMinMembers)
                
                # encode the groupID  
                spark_fof_c.encode_gid(part_arr, partition_index)
        
            for arr in np.split(part_arr, range(batch_size,len(part_arr),batch_size)):
                yield arr

        partitioned_rdd = self.partitioned_rdd

        fof_rdd = partitioned_rdd.mapPartitionsWithIndex(lambda index, particles: run_local_fof(index, particles, tau, 1)).cache()

        return fof_rdd


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
        fof_rdd = self.fof_rdd
        domain_containers = self.domain_containers
        sc = self.sc

        N_partitions = sc.defaultParallelism*20

        groups_map = (fof_rdd.flatMap(lambda p: p[np.where(p['is_ghost'])[0]])
                             .map(pid_gid)
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
        fof_rdd = self.fof_rdd
        domain_containers = self.domain_containers

        def remap_partition(particles, gmap):
            for p_arr in particles: 
                remap_gid_partition_cython(p_arr, gmap)
                yield p_arr

        for l in range(level, -1, -1):
            m = self.get_level_map(l)
            m_b = self.sc.broadcast(m)
            merged_rdd = fof_rdd.mapPartitions(lambda particles: remap_partition(particles, m_b.value))

        return merged_rdd


    def finalize_groups(self):
        """
        Produce a mapping of group IDs such that group IDs are in the 
        order of group size and relabel the particle groups

        Returns a list of relabeled group IDs and particle counts.
        """

        nMinMembers = self.nMinMembers

        # define helper functions
        def count_groups(particle_array):
            gs, counts = np.unique(particle_array['iGroup'], return_counts=True)
            count_dict = dict()
        
            for i in xrange(len(gs)): 
                if gs[i] in count_dict: 
                    # here we explicitly cast to integer because the marshal serializer 
                    # otherwise garbles the data
                    count_dict[long(gs[i])] += long(counts[i])
                else:
                    count_dict[long(gs[i])] = long(counts[i])
            return count_dict

        def combine_dicts(d1, d2): 
            for k,v in d2.iteritems(): 
                if k in d1: 
                    d1[k] += v
                else: 
                    d1[k] = v
            return d1

        def relabel_groups_wrapper(p_arr, groups_map): 
            relabel_groups(p_arr, groups_map)
            return p_arr            

        merged_rdd = self.merged_rdd

        # first, get rid of ghost particles
        no_ghosts_rdd = merged_rdd.map(lambda p: p[np.where(p['is_ghost'] != GHOST_PARTICLE_COPY)[0]])

        filtered_groups = no_ghosts_rdd.map(count_groups).treeReduce(combine_dicts, 4)

        # get the final group mapping by sorting groups by particle count
        groups_map = {}

        for i, (g,c) in enumerate(sorted(filtered_groups.items(), key = lambda (x,y): y, reverse=True)): 
            groups_map[g] = i+1

        final_fof_rdd = no_ghosts_rdd.map(lambda p_arr: relabel_groups_wrapper(p_arr, groups_map))

        groups_map_inv = {v:k for (k,v) in groups_map.iteritems()}

        self._groups = [(i,filtered_groups[groups_map_inv[i]]) for i in range(1,len(filtered_groups)+1) \
                        if filtered_groups[groups_map_inv[i]] >= nMinMembers]

        return final_fof_rdd


def setup_domain(N, tau, maxes, mins):
    D = DomainRectangle(maxes, mins, tau=tau)
    domain_containers = D.split_domain(max_N=N)
    for r in domain_containers:
        r.bin = get_rectangle_bin(r, D.mins, D.maxes, 2**N)

    domain_containers.sort(key=lambda x: x.bin)

    return domain_containers


# def partition_particles(particles, domain_containers, tau, mins, maxs):
#     """Copy particles in buffer areas to the partitions that will need them"""

#     N = domain_containers[0].N

#     trans = np.array([[-tau, 0, 0], [0,-tau, 0], [0, 0, -tau], [-tau, -tau, 0], [0, -tau, -tau], [-tau,-tau,-tau]])

#     for p in particles:
#         pos = p['pos']
#         my_bins = []
#         my_bins.append(get_bin_cython(pos, 2**N, mins, maxs))

#         my_rect = domain_containers[my_bins[0]]

#         if rect_buffer_zone_cython(pos,domain_containers):
#             # particle coordinates in single array
#            # coords = np.copy(pos)
#             # iterate through the transformations
#             for t in trans: 
# #                x,y,z = coords + t
#                 trans_bin = get_bin_cython(pos+t, 2**N, mins, maxs)
#                 if trans_bin not in my_bins and trans_bin > 0:
#                     my_bins.append(trans_bin)
#                     yield (trans_bin, p)

#         # return the first bin, i.e. the only non-ghost bin
#         yield (my_bins[0], p)


def get_bin(pos, nbins, mins, maxs):
    px, py, pz = pos
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
    return get_bin(point.astype(np.float32), nbins, mins, maxs)


def get_buffer_particles(partition, particles, domain_containers, level=0):
    """Produce the particles from the buffer regions"""
    my_rect = domain_containers[partition]

    # get to the correct level
    for i in range(level):
        my_rect = my_rect.parent

    for p in np.concatenate(list(particles)):
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

    # def in_buffer_zone(self, p):
    #     """Determine whether a particle is in the buffer zone"""
    #     return rect_buffer_zone_cython(p['pos'], self.mins, self.maxes, self.bufferRectangle.mins, self.bufferRectangle.maxes)
