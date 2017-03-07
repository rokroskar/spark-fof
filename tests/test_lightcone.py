import pytest
import sparkhpc
from sparkhpc import sparkjob
import numpy as np

#correct_nparticles = {(32,34): 587910375, (30,32): 594482071}
correct_nparticles = 594482071

# set up all parameters
path = '/cluster/home/roskarr/projects/euclid/2Tlc-final/'

# domain parameters
diff = np.float32(0.033068776)
global_min = -31*diff
global_max = 31*diff

dom_maxs = np.array([global_max]*3, dtype=np.float64)
dom_mins = np.array([global_min]*3, dtype=np.float64)

tau = 0.2/12600 # 0.2 times mean interparticle separation

ncores = 8
minblock = 30
maxblock = 32
nBins = 62

@pytest.fixture()
def lightcone_analyzer(sc_distributed):
    """TipsyFOFAnalyzer with nMinMembers = [1,8]"""
    import spark_fof
    import numpy as np

    lightcone_analyzer = spark_fof.spark_fof.LCFOFAnalyzer(sc, path, 
                                                           nMinMembers, 62, tau, 
                                                           dom_mins, dom_maxs, 
                                                           blockids=range(30,32), 
                                                           buffer_tau=tau*2)

    return lightcone_analyzer

@pytest.fixture()
def groups():
    return np.load('testdata/fof_30_32_min8_groups.npy')

def count_particles(rdd): 
    """Helper function to add up the lenghts of all the particle arrays"""
    return rdd.map(lambda arr: len(arr)).treeReduce(lambda a,b: a+b)

def test_lightcone_file_read(lightcone_analyzer):
    """Check that the correct number of particles are read"""
    assert(count_particles(lightcone_analyzer.particle_rdd) == correct_nparticles)

def test_group_count(lightcone_analyzer, groups): 
    """Check that we have the correct number of groups at the end"""
    assert(len(lightcone_analyzer.groups) == len(groups))

def test_detailed_particle_counts(lightcone_analyzer, groups):
    """Test that the number of groups per particle count is correct"""
    group_count_counts = np.bincount(groups[1:]) # skip group 0

    # get the count of group counts from the spark fof
    group_count_arr = np.array([y for x,y in tipsy_analyzer.groups.iteritems()])
    group_count_counts2 = np.bincount(group_count_arr)

    assert(np.all(group_count_counts2 == group_count_counts))
