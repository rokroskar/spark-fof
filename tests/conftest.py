import pytest
import sparkhpc
from sparkhpc import sparkjob
import numpy as np
import os

def pytest_addoption(parser):
    parser.addoption("--clusterid", action="store", default=None)


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
@pytest.mark.unit(scope='module')
def sc_distributed(request):
    """set up a spark context on the cluster"""
    import findspark
    findspark.init()
    import os
    import time

    ncores = 8

    os.environ['SPARK_DRIVER_MEMORY'] = '8G'
    
    clusterid = request.config.getoption('--clusterid')

    if clusterid is not None: 
        sj = sparkjob.LSFSparkJob(clusterid=int(clusterid))
    else:
        sj = sparkjob.LSFSparkJob(ncores=ncores,
                                  memory=12000,
                                  walltime='02:00', 
                                  template=os.path.join(os.path.dirname(os.path.abspath(__file__)),'../run_scripts/job.template'))
        sj.wait_to_start()
        time.sleep(30)

    sc = sparkhpc.start_spark(master=sj.master_url, spark_conf='../conf', 
                              profiling=False, executor_memory='6000M', 
                              graphframes_package='graphframes:graphframes:0.3.0-spark2.0-s_2.11')

    sc.setCheckpointDir('file:///cluster/home/roskarr/work/euclid')
    
    yield sc

    sc.stop()
    if clusterid is None: 
        sj.stop()

