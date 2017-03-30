import numpy as np
from math import floor, ceil
from functools import total_ordering
from collections import defaultdict
from itertools import izip 
import re
import time
import os
import gc 
import warnings

# initialize spark to load spark classes
import findspark
findspark.init()
import pyspark
from pyspark.accumulators import AccumulatorParam
from pyspark.sql import SQLContext, Row
from pyspark.storagelevel import StorageLevel 
try: 
    import graphframes
except ImportError: 
    warnings.warn('graphframes not loaded')

# local imports
import spark_fof_c
from spark_fof_c import  remap_gid_partition_cython, \
                         relabel_groups, \
                         ghost_mask, \
                         partition_ghosts, \
                         partition_array, \
                         count_groups_partition_cython, \
                         pdt       
from . import fof
from domain import setup_domain

PRIMARY_GHOST_PARTICLE = 1
GHOST_PARTICLE_COPY = 2 

def partition_helper(pid): 
    return pid

class FOFAnalyzer(object):
    def __init__(self, sc, particles, nMinMembers, nBins, tau, 
                 dom_mins=[-.5,-.5,-.5], dom_maxs=[.5,.5,.5], 
                 nPartitions=None, buffer_tau = None, symmetric=False, DEBUG=False, **kwargs):
        self.sc = sc
        self.nBins = nBins
        self.tau = tau
        self.dom_mins = dom_mins
        self.dom_maxs = dom_maxs
        self.nMinMembers = nMinMembers
        self.symmetric = symmetric
        self.DEBUG = DEBUG

        if buffer_tau is None: 
            buffer_tau = 2*tau
            self.buffer_tau = buffer_tau
        else: 
            self.buffer_tau = buffer_tau 
            
        domain_containers = setup_domain(nBins, buffer_tau, dom_mins, dom_maxs) 
        self.domain_containers = domain_containers

        # set up domain limit arrays
        self.N = domain_containers[0].N
        self.n_containers = len(domain_containers)
        self.container_mins = np.zeros((self.n_containers, 3))
        self.container_maxs = np.zeros((self.n_containers, 3))
        self.buff_mins = np.zeros((self.n_containers, 3))
        self.buff_maxs = np.zeros((self.n_containers, 3))

        for i in range(self.n_containers): 
            self.container_mins[i] = domain_containers[i].mins
            self.container_maxs[i] = domain_containers[i].maxes
            self.buff_mins[i] = domain_containers[i].bufferRectangle.mins
            self.buff_maxs[i] = domain_containers[i].bufferRectangle.maxes

        if nPartitions is None: 
            self.nPartitions = len(domain_containers)
        else: 
            self.nPartitions = nPartitions

        if isinstance(particles, str): 
            self.particle_rdd = self.read_data(particles, **kwargs)
        elif isinstance(particles, pyspark.rdd.RDD):
            self.particle_rdd = particles
        else: 
            raise RuntimeError('particles need to be either a filesystem location or a pyspark rdd')

        # set up RDD place-holders
        self._partitioned_rdd = None
        self._fof_rdd = None
        self._merged_rdd = None
        self._final_fof_rdd = None
        self._groups = None


    def read_data(self):
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

        N, buffer_tau, dom_mins, dom_maxs, symmetric = self.N, self.buffer_tau, \
                                                       self.dom_mins, self.dom_maxs, \
                                                       self.symmetric 
        def partition_helper(iterator):
            for arr in iterator: 
                res = function(arr,N,buffer_tau,symmetric,dom_mins,dom_maxs)
                for r in res: 
                    yield r
        return rdd.mapPartitions(partition_helper)


    def partition_particles(self): 
        """
        Partitions the particles for running local FOF
        """

        nPartitions = self.nPartitions
        N, tau, dom_mins, dom_maxs = self.N, self.tau, self.dom_mins, self.dom_maxs

        # mark the ghosts
        self.particle_rdd = self._set_ghost_mask(self.particle_rdd)
        
        
        ghosts_rdd = (self._partition_rdd(self.particle_rdd, partition_ghosts)
                          .partitionBy(nPartitions)
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

        def run_local_fof(partition_index, particle_iter, tau, nMinMembers, batch_size=1024*10240, DEBUG=False): 
            """Helper function to run FOF locally on the individual partitions"""
            part_arr = np.concatenate(list(particle_iter))
            if len(part_arr)>0:
                tin = time.time()
                t = time.localtime()
                print 'spark_fof: running local fof on {part} started at {t.tm_hour:02}:{t.tm_min:02}:{t.tm_sec:02}'.format(part=partition_index,t=t)
                # run fof
                tin = time.time()
                part_arr.sort(kind='mergesort', order='iOrder')
                print 'sorting took %f seconds'%(time.time()-tin)
                
                fof.run(part_arr, tau, nMinMembers)
                print 'spark_fof: fof on {part} finished in {seconds}'.format(part=partition_index, seconds=time.time()-tin)

                print 'particles in partition %d: %d'%(partition_index, len(part_arr))
                if DEBUG:
                    for i in range(100): 
                        print 'spark_fof DEBUG: %d %d'%(part_arr['iOrder'][i], part_arr['iGroup'][i])

                # encode the groupID  
                spark_fof_c.encode_gid(part_arr, partition_index)
                if DEBUG: print 'spark_fof DEBUG: total number of groups in partition %d: %d'%(partition_index, len(np.unique(part_arr['iGroup'])))
                
               

            for arr in np.split(part_arr, range(batch_size,len(part_arr),batch_size)):
                yield arr

        partitioned_rdd = self.partitioned_rdd

        fof_rdd = partitioned_rdd.mapPartitionsWithIndex(lambda index, particles: run_local_fof(index, particles, tau, 1))

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
        sc = self.sc

        nPartitions = sc.defaultParallelism*5

        groups_map = (fof_rdd.flatMap(lambda p: p[np.where(p['is_ghost'])[0]])
                             .map(pid_gid)
                             .groupByKey(nPartitions)
                             .values()
                             .filter(lambda x: len(x)>1)
                             .map(lambda x: sorted(x))
                             .flatMap(lambda gs: [(g, gs[0]) for g in gs[1:]]))

        return groups_map
 

    def _get_level_map(self):
        """
        Produce a group re-mapping across sub-domains. Connected groups are obtained by finding
        groups belonging to the same particles and linking them into a graph. Each node in a 
        connected sub-graph is mapped to the lowest group ID in the sub-graph. 
        """
        
        # get the initial group mapping across sub-domains just based on
        # particle IDs
        groups_map = self._get_gid_map()

        sc = self.sc

        sqc = SQLContext(sc)

        if self.DEBUG: 
            print 'spark_fof DEBUG: groups in initial mapping = %d'%groups_map.cache().count()

        
        # create the spark GraphFrame with group IDs as nodes and group connections as edges
        v_df = sqc.createDataFrame(groups_map.flatMap(lambda x: x)
                                             .distinct()
                                             .map(lambda v: Row(id=int(v))))
        e_df = sqc.createDataFrame(groups_map.map(lambda (s,d): Row(src=int(s), dst=int(d))))

        # persist the graph, allowing it to spill to disk if necessary
        g_graph = graphframes.GraphFrame(v_df, e_df).persist(StorageLevel.MEMORY_AND_DISK_SER)
        
        # generate mapping
        def make_mapping(items): 
            """Helper function to generate mappings to lowest node ID"""
            compid, nodes = items
            nodes = list(nodes)
            base_node = min(nodes)
            return [(node,base_node) for node in nodes if node != base_node]
        
        nPartitions = sc.defaultParallelism*5

        timein = time.time()
        mapping = (g_graph.connectedComponents()
                          .rdd.map(lambda row: (row.component, row.id))
                          .groupByKey(nPartitions)
                          .filter(lambda (k,v): len(v.data)>1)
                          .flatMap(make_mapping)
                          .collectAsMap())

        if self.DEBUG:
            print 'spark_fof DEBUG: groups in final mapping = %d'%len(mapping)
            # from pickle import dump
            # with open('mapping2.dump', 'w') as f: 
            #     dump(mapping, f, -1)

        print 'spark_fof <timing>: domain group mapping build took %f seconds'%(time.time()-timein)
        return mapping


    def _merge_groups(self):
        """
        For an RDD of particles, discover the groups connected across domain
        boundaries and remap to a lowest common group ID. 
        """
        fof_rdd = self.fof_rdd
       
        def remap_partition(particles, gmap):
            """Helper function to remap groups"""
            for p_arr in particles: 
                remap_gid_partition_cython(p_arr, gmap)
                yield p_arr

        m = self._get_level_map()
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
        from pyspark.sql import Row

        merged_rdd = self.merged_rdd
        sc = self.sc
        sqc = pyspark.sql.SQLContext(sc)

        nPartitions = sc.defaultParallelism*5

        nMinMembers = self.nMinMembers

        # we need to use the group merge map used in a previous step to see which 
        # groups are actually spread across domain boundaries
        group_merge_map = self.group_merge_map
        gr_map_inv = {v:k for (k,v) in group_merge_map.iteritems()}
        gr_map_inv_b = sc.broadcast(gr_map_inv)

        def count_groups_partition(particle_arrays, gr_map_inv_b, nMinMembers): 
            p_arr = np.concatenate(list(particle_arrays))
            del(particle_arrays)
            gs, counts = np.unique(p_arr['iGroup'], return_counts=True)
            del(p_arr)
            gc.collect()
            gr_map_inv = gr_map_inv_b.value
            return ((g,cnt) for g,cnt in izip(gs,counts) if (g in gr_map_inv) or (cnt >= nMinMembers))

        def count_groups(p):
            gs, counts = np.unique(p['iGroup'], return_counts=True)
            return ((g,cnt) for g,cnt in izip(gs,counts))

        def relabel_groups_wrapper(p_arr, groups_map): 
            relabel_groups(p_arr, groups_map)
            return p_arr            

        # first, get rid of ghost particles
        no_ghosts_rdd = merged_rdd.map(lambda p: p[np.where(p['is_ghost'] != GHOST_PARTICLE_COPY)[0]])

        # count up the number of particles in each group in each partition
        group_counts = no_ghosts_rdd.mapPartitions(lambda p_arrs: count_groups_partition_cython(p_arrs, gr_map_inv_b, nMinMembers)).cache()
        
        # merge the groups that reside in multiple domains
        merge_group_counts = (group_counts.filter(lambda (g,cnt): g in gr_map_inv_b.value)
                                          .reduceByKey(lambda a,b: a+b, nPartitions)
                                          .filter(lambda (g,cnt): cnt>=nMinMembers)).cache()

        if self.DEBUG:
            print 'spark_fof DEBUG: non-merge groups = %d merge groups = %d'%(group_counts.count(), merge_group_counts.count())        

        # combine the group counts
        print 'total groups: ', (group_counts.filter(lambda (gid,cnt): gid not in gr_map_inv_b.value) + merge_group_counts).count()
        total_group_counts = (group_counts.filter(lambda (gid,cnt): gid not in gr_map_inv_b.value) + merge_group_counts).collect()
        self.total_group_counts = total_group_counts
        
        # get the final group mapping by sorting groups by particle count
        timein = time.time()
        groups_map = {}
        self._groups = {}
        for i, (g,c) in enumerate(total_group_counts): 
            groups_map[g] = i+1
            self._groups[i+1] = c

        print 'spark_fof: Final group map build took %f seconds'%(time.time() - timein)
        groups_map_b = sc.broadcast(groups_map)

        final_fof_rdd = no_ghosts_rdd.map(lambda p_arr: relabel_groups_wrapper(p_arr, groups_map_b.value))

        return final_fof_rdd


    def get_bin(pos):
        nbins, mins, maxs = self.nBins, self.dom_mins, self.dom_maxs
        return spark_fof_c.get_bin_wrapper(pos, nbins, mins, maxs)


def pid_gid(p):
    """Map the particle to its pid and gid"""
    return (p['iOrder'], p['iGroup'])


class dictAdd(AccumulatorParam):
            def zero(self, value):
                return {i:0 for i in range(len(value))}
            def addInPlace(self, val1, val2): 
                for k, v in val2.iteritems(): 
                    val1[k] += v
                return val1


class TipsyFOFAnalyzer(FOFAnalyzer):
    
    def read_data(self, filename, chunksize = 2048): 
        """
        Read a tipsy file and set the sequential particle IDs
        
        This scans through the data twice -- first to convert the data to fof format
        and a second time to set the particle IDs.
        """
        pdt_tipsy = np.dtype([('mass', 'f4'),('pos', 'f4', 3),('vel', 'f4', 3), ('eps', 'f4'), ('phi', 'f4')])

        # helper functions
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

        rec_rdd = rec_rdd.mapPartitionsWithIndex(set_particle_IDs_partition)
        rec_rdd = (self._partition_rdd(rec_rdd, partition_array).partitionBy(self.nPartitions) 
                                                                .map(lambda (_,v): v, preservesPartitioning=True))  
        return rec_rdd


class LCFOFAnalyzer(FOFAnalyzer):
    """
    An extension of the FOFAnalyzer class designed specifically for the "light-cone" output. 

    This is using the light-cone data divided into a 62^3 grid (actually 64^3, but outer 
    edges are cut off). 
    """

    # these set up the grid 
    # diff is the width of each grid cell 
    # diff = np.float32(0.033068776)
    # global_min = -31*diff
    # global_max = 31*diff

    # # domain limits
    # dom_maxs = np.array([global_max]*3, dtype=np.float64)
    # dom_mins = np.array([global_min]*3, dtype=np.float64)

    # linking length and buffer region size
    #tau = diff*5./125.
    # tau = 0.2/12600
    # buffer_tau = diff*5./150.

    def __init__(self, sc, path, *args, **kwargs):
        self.path = path
        self._ids_map = None
        self._global_to_local_map = None
        self._local_to_global_map = None
        super(LCFOFAnalyzer, self).__init__(sc, path, *args, **kwargs)

    @property
    def global_to_local_map(self):
        """Function to map from file block numbers to domain bin"""
        Ngrid = 62
        map_file_to_domain = lambda (x,y,z): (x-1) + (y-1)*Ngrid + (z-1)*Ngrid*Ngrid

        if self._global_to_local_map is None: 
            m = {}
            for k,v in self.ids_map.iteritems(): 
                m[map_file_to_domain(k)] = v
            self._global_to_local_map = m
        return self._global_to_local_map


    @property
    def local_to_global_map(self):
        if self._local_to_global_map is None:
            m = {}
            for k,v in self.global_to_local_map.iteritems():
                m[v] = k
            self._local_to_global_map = m
        return self._local_to_global_map


    def read_data(self, path, **kwargs):
        """
        Read blocks found under `path`

        If `blockids` keyword is provided, only blocks in `blockids` will be read, 
        otherwise all files matching blk.X.Y.Zi will be read. 

        Parameters
        ----------

        sc: pyspark.SparkContext

        path: directory path

        blockids: list of block ids to read (optional)
        """

        from glob import glob
        sc = self.sc
        pdt_lc = np.dtype([('pos', 'f4', 3),('vel', 'f4', 3)])

        blockids = kwargs['blockids']

        def set_particle_IDs_partition(index, iterator): 
            """
            Use the aggregate partition counts to set monotonically increasing 
            particle indices
            """
            p_counts = partition_counts.value
            local_index = 0
            start_index = sum([p_counts[i] for i in range(index)])
            for arr in iterator:
                arr['iOrder'] = range(start_index + local_index, start_index + local_index + len(arr))
                arr['iGroup'] = loc_to_glob_map_b.value[index]
                local_index += len(arr)
                yield arr
        
        def read_file(index, i, chunksize=102400): 
            for part,filename in i:
                timein = time.time()
                with open(filename,'rb') as f: 
                    header = f.read(62500)
                    while True:
                        chunk = f.read(chunksize*24)
                        if len(chunk): 
                            p_arr = np.frombuffer(chunk, pdt_lc)
                            new_arr = np.zeros(len(p_arr), dtype=pdt)
                            new_arr['pos'] = p_arr['pos']
                            yield new_arr
                        else: 
                            print 'spark_fof: reading %s took %d seconds in partition %d'%(filename, time.time()-timein, index)
                            break
        
        # determine which files to read
        get_block_ids = re.compile('blk\.(\d+)\.(\d+)\.(\d+)?')

        if blockids is None: 
            files = glob(os.path.join(self.path,'*/*'))
        else: 
            files = []
            for dirname, subdirlist, filelist in os.walk(path):
                try: 
                    dirnum = int(os.path.basename(dirname))
                    if dirnum in blockids: 
                        for f in filelist:
                            ids = get_block_ids.findall(f)
                            if len(ids) > 0:
                                if all(int(x) in blockids for x in ids[0]):
                                    files.append(os.path.join(dirname,f))
                except ValueError: 
                    pass

        files.sort()
        nfiles = len(files) 
        self.nPartitions = nfiles

        print 'spark_fof: Number of input files: ', nfiles

        # get particle counts per partition
        nparts = {i:_get_nparts(filename,62500,pdt_lc.itemsize) for i,filename in enumerate(files)}

        print 'spark_fof: Total number of particles: ', np.array(nparts.values()).sum()
        
        # set up the map from x,y,z to partition id        
        ids = map(lambda x: tuple(map(int, get_block_ids.findall(x)[0])), files)
        ids_map = {x:i for i,x in enumerate(ids)}
        self.ids_map = ids_map
        loc_to_glob_map_b = self.local_to_global_map
        
        ids_map_b = sc.broadcast(ids_map)
        loc_to_glob_map_b = sc.broadcast(loc_to_glob_map_b)

        partition_counts = sc.broadcast(nparts)

        rec_rdd = (sc.parallelize(zip(ids,files), numSlices=self.nPartitions)
                     .map(lambda (id,filename): (ids_map_b.value[id],filename))
                     .partitionBy(self.nPartitions).cache()
                     .mapPartitionsWithIndex(read_file, preservesPartitioning=True)
                     .mapPartitionsWithIndex(set_particle_IDs_partition, 
                                                       preservesPartitioning=True))
      
        return rec_rdd

    def partition_particles(self): 
        """
        Partitions the particles for running local FOF
        """

        nPartitions = self.nPartitions
        N, tau, dom_mins, dom_maxs = self.N, self.tau, self.dom_mins, self.dom_maxs

        # mark the ghosts
        self.particle_rdd = self._set_ghost_mask(self.particle_rdd)
        
        gl_to_loc_map = self.global_to_local_map
        gl_to_loc_map_b = self.sc.broadcast(gl_to_loc_map)

        def remap_partition(particles):
            """Helper function to remap groups"""
            remap_gid_partition_cython(particles, gl_to_loc_map_b.value)
            return particles

        ghosts_rdd = (self._partition_rdd(self.particle_rdd, partition_ghosts)
                          .filter(lambda (k,v): k in gl_to_loc_map_b.value)
                          .map(lambda (k,v): (gl_to_loc_map_b.value[k],v))
                          .partitionBy(nPartitions)
                          .map(lambda (k,v): v, preservesPartitioning=True))
    
        part_rdd = self.particle_rdd

        partitioned_rdd = ghosts_rdd + part_rdd
        self._partitioned_rdd = partitioned_rdd

        return partitioned_rdd

def _get_nparts(filename,headersize,itemsize): 
    """Helper function to get the number of particles in the file"""
    return (os.path.getsize(filename)-headersize)/itemsize

