import numpy as np
from math import floor, ceil
from functools import total_ordering
import networkx as nx
from collections import defaultdict
from itertools import izip 

# initialize spark to load spark classes
import findspark
findspark.init()
import pyspark
from pyspark.accumulators import AccumulatorParam
from . import spark_tipsy

# local imports
import spark_fof_c
from spark_fof_c import  remap_gid_partition_cython, \
                         relabel_groups, \
                         ghost_mask, \
                         partition_ghosts, \
                         partition_array, \
                         pdt       
from . import fof
from domain import setup_domain

PRIMARY_GHOST_PARTICLE = 1
GHOST_PARTICLE_COPY = 2 

def partition_helper(pid): 
    return pid

class FOFAnalyzer():
    def __init__(self, sc, particles, 
                 nMinMembers, nBins, tau, 
                 dom_mins=[-.5,-.5,-.5], dom_maxs=[.5,.5,.5], 
                 Npartitions=None, buffer_tau = None, symmetric=False):
        self.sc = sc
        self.nBins = nBins
        self.tau = tau
        self.dom_mins = dom_mins
        self.dom_maxs = dom_maxs
        self.nMinMembers = nMinMembers
        self.symmetric = symmetric

        if buffer_tau is None: 
            buffer_tau = tau

        domain_containers = setup_domain(nBins, buffer_tau, dom_mins, dom_maxs) 
        self.domain_containers = domain_containers

        # set up domain limit arrays
        self.N = domain_containers[0].N
        self.n_containers = len(domain_containers)
        self.container_mins = np.zeros((self.n_containers, 3))
        self.container_maxs = np.zeros((self.n_containers, 3))
        self.buff_mins = np.zeros((self.n_containers, 3))
        self.buff_maxs = np.zeros((self.n_containers, 3))

        self.domain_containers_b = self.sc.broadcast(domain_containers)

        for i in range(self.n_containers): 
            self.container_mins[i] = domain_containers[i].mins
            self.container_maxs[i] = domain_containers[i].maxes
            self.buff_mins[i] = domain_containers[i].bufferRectangle.mins
            self.buff_maxs[i] = domain_containers[i].bufferRectangle.maxes

        if Npartitions is None: 
            self.Npartitions = len(domain_containers)
        else: 
            self.Npartitions = Npartitions

        if isinstance(particles, str): 
            # we assume we have a file location
            p_rdd = spark_tipsy.read_tipsy_output(sc, particles, chunksize=1024*4)
            self.particle_rdd = (self._partition_rdd(p_rdd, partition_array) 
                                     .partitionBy(self.Npartitions) 
                                     .map(lambda (_,v): v, preservesPartitioning=True))
        
        elif isinstance(particles, pyspark.rdd.RDD):
            self.particle_rdd = particles

        # set up RDD place-holders
        self._partitioned_rdd = None
        self._fof_rdd = None
        self._merged_rdd = None
        self._final_fof_rdd = None
        self._groups = None

        self.global_to_local_map = None


    def read_file(self):
        raise NotImplementedError()

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
            self._merged_rdd = self._merge_groups()
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
        return self.final_fof_rdd


    def _set_ghost_mask(self, rdd):
        """
        Set the `is_ghost` flag for the particle arrays in the rdd.

        Parameters
        ----------

        rdd : pyspark.RDD
            the RDD of particle arrays to update with a ghost mask
        """
        tau, N, dom_mins, dom_maxs = self.tau, self.N, self.dom_mins, self.dom_maxs
        container_mins, container_maxs = self.container_mins, self.container_maxs
        buff_mins, buff_maxs = self.buff_mins, self.buff_maxs

        def ghost_map_wrapper(iterator): 
            for arr in iterator: 
                ghost_mask(arr, tau, N, container_mins, container_maxs, 
                           buff_mins, buff_maxs, dom_mins, dom_maxs)
                yield arr

        return rdd.mapPartitions(ghost_map_wrapper, preservesPartitioning=True)
        

    def _partition_rdd(self, rdd, function):
        """
        Helper function for setting up the arrays for partitioning

        Parameters
        ----------

        rdd : pyspark.RDD
            the RDD of particle arrays to repartition
        function : python function
            the function to use that will yield tuples of (group_id, particle_array) for
            repartitioning - see `spark_fof.spark_fof_c.partition_ghosts`
        """ 

        N, tau, dom_mins, dom_maxs, symmetric = self.N, self.tau, self.dom_mins, self.dom_maxs, self.symmetric
        def partition_helper(iterator):
            for arr in iterator: 
                res = function(arr,N,tau,symmetric,dom_mins,dom_maxs)
                for r in res: 
                    yield r
        return rdd.mapPartitions(partition_helper)


    def partition_particles(self): 
        """
        Partitions the particles for running local FOF
        """

        Npartitions = self.Npartitions
        N, tau, dom_mins, dom_maxs = self.N, self.tau, self.dom_mins, self.dom_maxs

        # mark the ghosts
        self.particle_rdd = self._set_ghost_mask(self.particle_rdd)
        
        if self.global_to_local_map is not None: 
            gl_to_loc_map = self.global_to_local_map
            gl_to_loc_map_b = self.sc.broadcast(gl_to_loc_map)

            def remap_partition(particles):
                """Helper function to remap groups"""
                remap_gid_partition_cython(particles, gl_to_loc_map_b.value)
                return particles

            ghosts_rdd = (self._partition_rdd(self.particle_rdd, partition_ghosts)
                              .filter(lambda (k,v): k in gl_to_loc_map_b.value)
                              .map(lambda (k,v): (gl_to_loc_map_b.value[k],v))
                              .partitionBy(Npartitions)
                              .map(lambda (k,v): remap_partition(v), preservesPartitioning=True))
        else:
            ghosts_rdd = (self._partition_rdd(self.particle_rdd, partition_ghosts)
                              .partitionBy(Npartitions)
                              .map(lambda (_,v): v, preservesPartitioning=True))

        part_rdd = self.particle_rdd
        partitioned_rdd = ghosts_rdd + part_rdd
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


    def _get_gid_map(self, level=0):
        """
        Take a particle RDD and return a gid -> gid' that will link groups in the buffer region.

        This is done in two steps: 
            - first, a mapping of buffer region particle IDs to group IDs is made
            - second, the particle IDs corresponsing to the buffer region particles
              are filtered from the full data and a map is produced that maps all groups
              onto the group corresponding to the lowest container ID

        Returns:

        list of tuples of (src,dst) group ID mappings 
        """
        fof_rdd = self.fof_rdd
        domain_containers = self.domain_containers
        sc = self.sc

        N_partitions = sc.defaultParallelism*10

        groups_map = (fof_rdd.flatMap(lambda p: p[np.where(p['is_ghost'])[0]])
                             .map(pid_gid)
                             .groupByKey(N_partitions)
                             .values()
                             .map(lambda x: sorted(x))
                             .flatMap(lambda gs: [(g, gs[0]) for g in gs[1:]])).collect()

        return groups_map
 
    def _get_level_map(self, level=0):
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
        groups_map = self._get_gid_map(level)

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

    def _merge_groups(self, level=0):
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
            """Helper function to remap groups"""
            for p_arr in particles: 
                remap_gid_partition_cython(p_arr, gmap)
                yield p_arr

        for l in range(level, -1, -1):
            m = self._get_level_map(l)
            m_b = self.sc.broadcast(m)
            merged_rdd = fof_rdd.mapPartitions(lambda particles: remap_partition(particles, m_b.value))

        self.group_merge_map = m

        return merged_rdd

    def finalize_groups(self):
        """
        Produce a mapping of group IDs such that group IDs are in the 
        order of group size and relabel the particle groups

        Returns a list of relabeled group IDs and particle counts.
        """

        nMinMembers = self.nMinMembers

        def count_groups_partition(particle_arrays, gr_map_inv_b, nMinMembers): 
            p_arr = np.concatenate(list(particle_arrays))
            gs, counts = np.unique(p_arr['iGroup'], return_counts=True)
            gr_map_inv = gr_map_inv_b.value
            return ((g,cnt) for g,cnt in izip(gs,counts) if (g in gr_map_inv) or (cnt >= nMinMembers))

        def relabel_groups_wrapper(p_arr, groups_map): 
            relabel_groups(p_arr, groups_map)
            return p_arr            

        merged_rdd = self.merged_rdd
        sc = self.sc

        # we need to use the group merge map used in a previous step to see which 
        # groups are actually spread across domain boundaries
        group_merge_map = self.group_merge_map
        gr_map_inv = {v:k for (k,v) in group_merge_map.iteritems()}
        gr_map_inv_b = sc.broadcast(gr_map_inv)

        # first, get rid of ghost particles
        no_ghosts_rdd = merged_rdd.map(lambda p: p[np.where(p['is_ghost'] != GHOST_PARTICLE_COPY)[0]])

        # count up the number of particles in each group in each partition
        group_counts = no_ghosts_rdd.mapPartitions(lambda p_arrs: count_groups_partition(p_arrs, gr_map_inv_b, nMinMembers))

        # merge the groups that reside in multiple domains
        merge_group_counts = (group_counts.filter(lambda (g,cnt): g in gr_map_inv_b.value)
                                          .reduceByKey(lambda a,b: a+b)
                                          .filter(lambda (g,cnt): cnt>=nMinMembers))

        # combine the group counts
        total_group_counts = (group_counts.filter(lambda (gid,cnt): gid not in gr_map_inv_b.value) + merge_group_counts).collect()
        self.total_group_counts = total_group_counts
        
        # get the final group mapping by sorting groups by particle count
        groups_map = {}
        self._groups = {}
        for i, (g,c) in enumerate(total_group_counts): 
            groups_map[g] = i+1
            self._groups[i] = c

        final_fof_rdd = no_ghosts_rdd.map(lambda p_arr: relabel_groups_wrapper(p_arr, groups_map))

        return final_fof_rdd


def get_bin(pos, nbins, mins, maxs):
    return spark_fof_c.get_bin_wrapper(pos, nbins, mins, maxs)


def pid_gid(p):
    """
    Map the particle to its pid and gid

    Parameters
    ----------

    p : single element of a numpy array with type `spark_fof_c.pdt`
    """
    return (p['iOrder'], p['iGroup'])


class TipsyFOFAnalyzer(FOFAnalyzer):
    
    def read_file(self, filename, chunksize = 2048): 
        """
        Read a tipsy file and set the sequential particle IDs
        
        This scans through the data twice -- first to convert the data to fof format
        and a second time to set the particle IDs.
        """
        pdt_tipsy = np.dtype([('mass', 'f4'),('pos', 'f4', 3),('vel', 'f4', 3), ('eps', 'f4'), ('phi', 'f4')])

        # helper functions
        class dictAdd(AccumulatorParam):
            def zero(self, value):
                return {i:0 for i in range(len(value))}
            def addInPlace(self, val1, val2): 
                for k, v in val2.iteritems(): 
                    val1[k] += v
                return val1

        def convert_to_fof_particle_partition(index, iterator): 
            for s in iterator: 
                p_arr = np.frombuffer(s, pdt_tipsy)
                new_arr = np.zeros(len(p_arr), dtype=pdt)
                new_arr['pos'] = p_arr['pos']  
                if count: 
                    npart_acc.add({index: len(new_arr)})
                yield new_arr

        def set_particle_IDs_partition(index, iterator): 
            p_counts = partition_counts.value
            local_index = 0
            start_index = sum([p_counts[i] for i in range(index)])
            for arr in iterator:
                arr['iOrder'] = range(start_index + local_index, start_index + local_index + len(arr))
                local_index += len(arr)
                yield arr
        
        sc = self.sc

        rec_rdd = sc.binaryRecords(filename, pdt_tipsy.itemsize*chunksize)
        nPartitions = rec_rdd.getNumPartitions()
        # set the partition count accumulator
        npart_acc = sc.accumulator({i:0 for i in range(nPartitions)}, dictAdd())
        count=True
        # read the data and count the particles per partition
        rec_rdd = rec_rdd.mapPartitionsWithIndex(convert_to_fof_particle_partition)
        rec_rdd.count()
        count=False

        partition_counts = sc.broadcast(npart_acc.value)

        return rec_rdd.mapPartitionsWithIndex(set_particle_IDs_partition)



class LCFOFAnalyzer(FOFAnalyzer):
    pass


