#!/bin/env python
#BSUB -J single-core-fof
#BSUB -W 02:00 
#BSUB -o logs/single-core-fof-%J.log
#BSUB -n 1
#BSUB -R rusage[mem=24000]

import os, re
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
from spark_fof.spark_fof_c import pdt

# set up all parameters
path = '/cluster/home/roskarr/projects/euclid/2Tlc-final/'

# domain parameters
diff = np.float32(0.033068776)
global_min = -31*diff
global_max = 31*diff

dom_maxs = np.array([global_max]*3, dtype=np.float64)
dom_mins = np.array([global_min]*3, dtype=np.float64)

tau = 0.2/12600 # 0.2 times mean interparticle separation
buffer_tau = diff*5./150.

minblock = 30
maxblock = 32

# run fof
import spark_fof
from itertools import izip 

def read_file(i, chunksize=102400): 
    """This reads an iterator of (partition_ID, filename) and returns particle arrays"""
    pdt_lc = np.dtype([('pos', 'f4', 3),('vel', 'f4', 3)])

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
                    print 'spark_fof: reading %s took %d seconds'%(filename, time.time()-timein)
                    break
                    

timein = time.time()
t = time.localtime()
print '--------------------'
print 'Starting fof at {t.tm_hour:02}:{t.tm_min:02}:{t.tm_sec:02}'.format(t=t)
print '--------------------'

timein2 = time.time()
blockids = range(30,32)

# determine which files to read
get_block_ids = re.compile('blk\.(\d+)\.(\d+)\.(\d+)i')

if blockids is None: 
    files = glob(os.path.join(self.path,'*/*'))
else: 
    files = []
    for dirname, subdirlist, filelist in os.walk(path):
        for f in filelist:
            ids = get_block_ids.findall(f)[0]
            if all(int(x) in blockids for x in ids):
                files.append(os.path.join(dirname,f))

print len(files)
files.sort()

p_arr = np.concatenate(list(read_file(izip(range(len(files)), files))))
p_arr['iOrder'] = np.linspace(0,len(p_arr)-1, len(p_arr), dtype=np.int32)

print 'data read took %f seconds'%(time.time() - timein2)
print 'number of particles: %d'%len(p_arr)

from spark_fof.fof import fof
timein2 = time.time()
fof.run(p_arr,tau,8)
print 'fof finished in %f seconds'%(time.time()-timein2)

np.save('fof_8_blocks4', p_arr)


# t = time.localtime()
# print '--------------------'
# print 'fof finished at {t.tm_hour:02}:{t.tm_min:02}:{t.tm_sec:02}'.format(t=t)
# print 'Number of groups: %d'%ngroups
# print 'cores: %d\tblocks: %d\ttime elapsed: %f'%(ncores, (maxblock-minblock)**3, time.time()-timein)


# print 'Printing memory info'
# def get_executor_data(x):
#     tempdir = os.path.join(os.environ['__LSF_JOB_TMPDIR__'],'work')
#     found = False
#     lines = []
#     for dirname, subdirlist, filelist in os.walk(tempdir):
#         for filename in filelist:
#             if filename == 'stderr':
#                 found = True
#                 print 'found log file'
#                 with open(os.path.join(dirname,filename),'r') as f:
#                     for line in f.readlines():
#                         if line.startswith('spark_fof:'):
#                             lines.append(line)
#                             print line
#     if not found:
#         lines = ['stderr file not found',]
#     return lines

# for i, res in enumerate(sc.parallelize(range(ncores)).flatMap(get_executor_data).collect()):
#     print res

# print 'Printing memory info done'
# sc.stop()
# sj.stop()
