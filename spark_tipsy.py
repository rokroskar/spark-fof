import numpy as np

from pyspark.accumulators import AccumulatorParam

#
# Code for reading in a (headerless) tipsy file
#

class dictAdd(AccumulatorParam):
    def zero(self, value):
        return {i:0 for i in range(len(value))}
    def addInPlace(self, val1, val2): 
        for k, v in val2.iteritems(): 
            val1[k] += v
        return val1

# define the data type
pdt_tipsy = np.dtype([('mass', 'f4'),('pos', 'f4', 3),('vel', 'f4', 3), ('eps', 'f4'), ('phi', 'f4')])

from spark_fof import pdt

def read_tipsy_output(sc, filename, chunksize = 2048): 
    """
    Read a tipsy file and set the sequential particle IDs
    
    This scans through the data twice -- first to get partition particle counts
    and a second time to actually set the particle IDs.
    """
    
    # helper functions
    def convert_to_fof_particle(s): 
        p_arr = np.frombuffer(s, pdt_tipsy)

        new_arr = np.zeros(len(p_arr), dtype=pdt)
        new_arr['pos'] = p_arr['pos']    
        return new_arr

    def convert_to_fof_particle_partition(index, iterator): 
        for s in iterator: 
            a = convert_to_fof_particle(s)
            if count: 
                npart_acc.add({index: len(a)})
            yield a

    def set_particle_IDs_partition(index, iterator): 
        p_counts = partition_counts.value
        local_index = 0
        start_index = sum([p_counts[i] for i in range(index)])
        for arr in iterator:
            arr['iOrder'] = range(start_index + local_index, start_index + local_index + len(arr))
            local_index += len(arr)
            yield arr
    
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

