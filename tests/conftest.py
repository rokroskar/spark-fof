import pytest
import sparkhpc
from sparkhpc import sparkjob
import numpy as np


@pytest.fixture(scope='module')
def sc_local():
    """set up a local spark context"""
    import findspark
    findspark.init()
    import os
    import time

    os.environ['SPARK_DRIVER_MEMORY'] = '8G'
    
    time.sleep(30)

    sc = sparkhpc.start_spark(master='local[4]')
    sc.setCheckpointDir('file:///cluster/home/roskarr/work/euclid')
    
    yield sc

    sc.stop()


@pytest.fixture(scope='module')
def sc_distributed():
    """set up a spark context on the cluster"""
    import findspark
    findspark.init()
    import os
    import time

    ncores = 8

    os.environ['SPARK_DRIVER_MEMORY'] = '8G'
    
    sj = sparkjob.LSFSparkJob(ncores=ncores,memory=12000,walltime='02:00', template='../run_scripts/job.template')

    sj.wait_to_start()

    time.sleep(30)

    sc = sparkhpc.start_spark(master=sj.master_url, spark_conf='../conf', 
                              profiling=False, executor_memory='6000M', 
                              graphframes_package='graphframes:graphframes:0.3.0-spark2.0-s_2.11')

    sc.setCheckpointDir('file:///cluster/home/roskarr/work/euclid')
    
    yield sc

    sc.stop()
