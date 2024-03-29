#!/bin/env python
#SBATCH -J spark-fof-driver
#SBATCH -t 72:00:00
#SBATCH -o spark-fof-driver-%J.log
#SBATCH -n 1
#SBATCH --mem-per-cpu=20000
#SBATCH -x x09y[01-12]

print 'RUNNING SPARK_FOF'

import os
#os.environ['SPARK_HOME'] = os.path.join(os.path.expanduser('~'), 'spark')
os.environ['JAVA_HOME'] = '/home/ics/roskar/data/src/jdk1.8.0_131'
os.environ['SPARK_HOME'] = '/home/ics/roskar/data/src/spark2'
import findspark
findspark.init()

os.environ['SPARK_CONF_DIR'] = '../conf'
os.environ['SPARK_DRIVER_MEMORY'] = '20G'

import pyspark
from pyspark import SparkContext, SparkConf
import sparkhpc
import time
import numpy as np


# set up all parameters
#path = '/cluster/home/roskarr/projects/euclid/2Tlc-final/'
path = '/zbox/trove/euclid/2Tlc-final/'

# domain parameters
diff = np.float32(0.03306878)
global_min = -31*diff
global_max = 31*diff

dom_maxs = np.array([global_max]*3, dtype=np.float64)
dom_mins = np.array([global_min]*3, dtype=np.float64)

tau = 0.2/12600 # 0.2 times mean interparticle separation

memory_per_node = 64000
cores_per_executor = 8
memory_per_core = memory_per_node/cores_per_executor - memory_per_node/cores_per_executor%1000
ncores = 576
minblock = 18
maxblock = 42

# submit sparkjob
sj = sparkhpc.sparkjob.sparkjob(ncores=ncores,
                                cores_per_executor=cores_per_executor, 
                                memory_per_executor=16000, 
                                memory_per_core=memory_per_core,
                                walltime='72:00',
                                template='sparkjob.slurm.template')

sj.wait_to_start()

print 'cluster started'
# wait for the job to get set up
time.sleep(30)

# initialize sparkContext
sc = sj.start_spark(spark_conf='../conf')

sc.setCheckpointDir('file:///zbox/data/roskar/checkpoint')
#sc.setCheckpointDir('file:///cluster/home/roskarr/work/euclid')

timeout = 300
timein = time.time()
while(sc.defaultParallelism < ncores): 
    time.sleep(2)
    if time.time() - timein > timeout:
        sc.stop()
        sj.stop()
        raise RuntimeError('Only %d cores out of %d requested were started before timeout'%(sc.defaultParallelism, ncores))

# run fof
import spark_fof

timein = time.time()
t = time.localtime()
print '--------------------'
print 'Starting spark-fof at {t.tm_hour:02}:{t.tm_min:02}:{t.tm_sec:02}'.format(t=t)
print '--------------------'

nMinMembers = 8
nBins = 62
fof_analyzer = spark_fof.spark_fof.LCFOFAnalyzer(sc, path, nMinMembers, nBins, tau, dom_mins, dom_maxs, blockids=range(minblock,maxblock), buffer_tau=tau*2)

fof_analyzer.finalize_groups()

t = time.localtime()
print '--------------------'
print 'spark-fof finished at {t.tm_hour:02}:{t.tm_min:02}:{t.tm_sec:02}'.format(t=t)
print 'cores: %d\tblocks: %d\ttime elapsed: %f'%(ncores, (maxblock-minblock)**3, time.time()-timein)

sc.stop()
sj.stop()
