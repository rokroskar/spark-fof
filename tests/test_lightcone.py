import pytest
import sparkhpc
from sparkhpc import sparkjob
import numpy as np

# set up all parameters
path = '/cluster/home/roskarr/projects/euclid/2Tlc-final/'

# domain parameters
diff = np.float32(0.03306878)
global_min = -31*diff
global_max = 31*diff

dom_maxs = np.array([global_max]*3, dtype=np.float64)
dom_mins = np.array([global_min]*3, dtype=np.float64)

tau = 0.2/12600 # 0.2 times mean interparticle separation

ncores = 8
minblock = 30
maxblock = 32
nBins = 62
nMinMembers = 8

@pytest.fixture()
def lightcone_analyzer(sc_distributed):
    """LCFOFAnalyzer"""
    import spark_fof
    import numpy as np

    lightcone_analyzer = spark_fof.spark_fof.LCFOFAnalyzer(sc_distributed, path, 
                                                           nMinMembers, 62, tau, 
                                                           dom_mins, dom_maxs, 
                                                           blockids=range(30,32), 
                                                           buffer_tau=tau*2)

    return lightcone_analyzer

@pytest.fixture()
def group_counts():
    return np.load('testdata/fof_30_32_min8_groups.npy')

def count_particles(rdd): 
    """Helper function to add up the lenghts of all the particle arrays"""
    return rdd.map(lambda arr: len(arr)).treeReduce(lambda a,b: a+b)

def test_lightcone_file_read(lightcone_analyzer, group_counts):
    """Check that the correct number of particles are read"""
    assert(count_particles(lightcone_analyzer.particle_rdd) == group_counts.sum())

def test_group_count(lightcone_analyzer, group_counts): 
    """Check that we have the correct number of groups at the end"""
    assert(len(lightcone_analyzer.groups) == len(group_counts)-1)

def test_detailed_particle_counts(lightcone_analyzer, group_counts):
    """Test that the number of groups per particle count is correct"""
    group_count_counts = np.bincount(group_counts[1:]) # skip group 0

    # get the count of group counts from the spark fof
    group_count_arr = np.array([y for x,y in lightcone_analyzer.groups.iteritems()])
    group_count_counts2 = np.bincount(group_count_arr)

    assert(np.all(group_count_counts2 == group_count_counts))

def test_boundary_ghost_particles(lightcone_analyzer):
    """Make sure that particles at the boundaries are correctly ghosted"""
    ps = lightcone_analyzer.partitioned_rdd.flatMap(lambda p: p[np.where(p['iOrder'] == 458000793)]).collect()

    # first make sure that there is a ghost
    assert(len(ps) == 2)

    # now check that it is in the correct partition

