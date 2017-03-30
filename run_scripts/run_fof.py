#!/bin/env python
#BSUB -J spark-fof-driver
#BSUB -W 1:00 
#BSUB -o logs/spark-fof-driver-%J.log
#BSUB -n 1
#BSUB -R rusage[mem=16000]

import os
os.environ['SPARK_HOME'] = os.path.join(os.path.expanduser('~'), 'spark')
import findspark
findspark.init()

os.environ['SPARK_CONF_DIR'] = '../conf'
os.environ['SPARK_DRIVER_MEMORY'] = '8G'

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

ncores = 1000
minblock = 20
maxblock = 40

# submit sparkjob
sj = sparkhpc.sparkjob.LSFSparkJob(ncores=ncores,memory=50000,walltime='24:00', template='../notebooks/sparkjob.slurm.template')
sj.wait_to_start()

# wait for the job to get set up
#time.sleep(30)

# initialize sparkContext
sc = sparkhpc.start_spark(master=sj.host_url, spark_conf='../conf', 
                          profiling=False, executor_memory='30000M', graphframes_package='graphframes:graphframes:0.3.0-spark2.0-s_2.11')

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
ngroups = len(fof_analyzer.groups)

t = time.localtime()
print '--------------------'
print 'spark-fof finished at {t.tm_hour:02}:{t.tm_min:02}:{t.tm_sec:02}'.format(t=t)
print 'Number of groups: %d'%ngroups
print 'cores: %d\tblocks: %d\ttime elapsed: %f'%(ncores, (maxblock-minblock)**3, time.time()-timein)

sc.stop()
sj.stop()
