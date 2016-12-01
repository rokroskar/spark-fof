import pytest
import sparkhpc
from sparkhpc import sparkjob
import numpy as np

correct_groups = {1: 7251094, 8: 105761, 64:13123}
correct_nparticles = 16777216

ghost_counts = {2:
                [(0, 1909852, 7381, 13241),
                 (1, 1734922, 10323, 5228),
                 (2, 2521632, 8968, 6142),
                 (3, 1984863, 7532, 2385),
                 (4, 2413127, 10824, 9872),
                 (5, 2238001, 11675, 2366),
                 (6, 1904376, 11839, 2899),
                 (7, 1994260, 7641, 0)],
                4:
                [(0, 401761, 2598, 4014),
                 (1, 122593, 1173, 1076),
                 (2, 137567, 1110, 1451),
                 (3, 270753, 2576, 1345),
                 (4, 232952, 1876, 1591),
                 (5, 282274, 1975, 3661),
                 (6, 242337, 2637, 1792),
                 (7, 182572, 1835, 1101),
                 (8, 281959, 2364, 2850),
                 (9, 269932, 2713, 3748),
                 (10, 131614, 1968, 1431),
                 (11, 178797, 2003, 875),
                 (12, 415761, 2817, 4268),
                 (13, 208986, 1975, 1109),
                 (14, 206767, 1528, 1190),
                 (15, 495302, 1784, 1067),
                 (16, 238094, 3741, 2108),
                 (17, 193218, 1698, 2855),
                 (18, 184583, 2213, 2568),
                 (19, 326342, 2503, 2137),
                 (20, 169711, 1405, 3122),
                 (21, 260863, 1301, 3266),
                 (22, 204843, 2579, 1497),
                 (23, 178831, 1964, 572),
                 (24, 408884, 2970, 2497),
                 (25, 151567, 2612, 4141),
                 (26, 198425, 2129, 2728),
                 (27, 154035, 1102, 1359),
                 (28, 291150, 5641, 2490),
                 (29, 476639, 4630, 1869),
                 (30, 346141, 2142, 1367),
                 (31, 265778, 2880, 418),
                 (32, 280470, 2552, 1863),
                 (33, 291521, 2846, 4767),
                 (34, 497947, 4685, 3247),
                 (35, 321844, 2837, 2920),
                 (36, 432388, 2711, 3291),
                 (37, 262652, 3955, 3690),
                 (38, 216331, 3069, 1380),
                 (39, 183105, 1225, 904),
                 (40, 153334, 1388, 2622),
                 (41, 275733, 4338, 1837),
                 (42, 208491, 2345, 3434),
                 (43, 135988, 1292, 1109),
                 (44, 197584, 1669, 2173),
                 (45, 367692, 3356, 1761),
                 (46, 343450, 2623, 1029),
                 (47, 262924, 1386, 875),
                 (48, 324303, 1971, 1761),
                 (49, 188525, 2087, 1137),
                 (50, 229591, 2199, 1026),
                 (51, 279037, 3519, 551),
                 (52, 342855, 3100, 1682),
                 (53, 280399, 1617, 1747),
                 (54, 234566, 2188, 1503),
                 (55, 265620, 1913, 870),
                 (56, 252096, 4676, 2064),
                 (57, 278119, 2145, 1703),
                 (58, 291151, 3201, 2427),
                 (59, 351294, 3299, 505),
                 (60, 156965, 2057, 1299),
                 (61, 212068, 2994, 317),
                 (62, 151559, 1465, 748),
                 (63, 239310, 2123, 0)]}


tau = 7.8125e-4
mins = np.array([-.5,-.5,-.5])
maxs = np.array([.5,.5,.5])
N = 2
filename = '/cluster/home/roskarr/work/euclid-test-files/euclid256.nat_no_header'


@pytest.fixture(scope='module')
def sc():
    import findspark
    findspark.init()
    from pyspark import SparkContext
    import os

    os.environ['SPARK_DRIVER_MEMORY'] = '8G'
    sc = sparkhpc.start_spark(master="local[2]")

    yield sc

    sc.stop()


@pytest.fixture(scope='module', params=[1,8])
def tipsy_analyzer(sc, request):
    """TipsyFOFAnalyzer with nMinMembers = [1,8]"""
    import spark_fof
    import numpy as np

    tipsy_analyzer = spark_fof.spark_fof.TipsyFOFAnalyzer(sc, filename, request.param, N, tau, mins, maxs)

    return tipsy_analyzer


@pytest.fixture(scope='module')
def tipsy_analyzer_single(sc, request):
    """TipsyFOFAnalyzer with nMinMembers = 8"""
    import spark_fof
    import numpy as np

    nMinMembers = 8
    tipsy_analyzer = spark_fof.spark_fof.TipsyFOFAnalyzer(sc, filename, nMinMembers, N, tau, mins, maxs)

    return tipsy_analyzer


def count_particles(rdd): 
    """Helper function to add up the lenghts of all the particle arrays"""
    return rdd.map(lambda arr: len(arr)).treeReduce(lambda a,b: a+b)


def count_ghosts(rdd):
    """Helper function to count up ghost particles per partition"""
    def count_ghosts_helper(index, iterator): 
        nghosts = 0
        nghosts_copy = 0
        nother = 0
        for arr in iterator: 
            nghosts += len(np.where(arr['is_ghost']==1)[0])
            nghosts_copy += len(np.where(arr['is_ghost']==2)[0])
            nother += len(np.where(arr['is_ghost']==0)[0])
        yield index,nother,nghosts,nghosts_copy
    return rdd.mapPartitionsWithIndex(count_ghosts_helper).collect()


def test_tipsy_file_read(tipsy_analyzer_single):
    """Check that the correct number of particles are read"""
    assert(count_particles(tipsy_analyzer_single.particle_rdd) == correct_nparticles)


def test_ghost_partitioning(tipsy_analyzer_single): 
    """Check that the ghost particles are correctly labeled and partitioned"""
    assert(count_ghosts(tipsy_analyzer_single.partitioned_rdd) == ghost_counts[int(tipsy_analyzer_single.N)])


def test_group_count(tipsy_analyzer): 
    """Check that we have the correct number of groups at the end"""
    assert(len(tipsy_analyzer.groups) == correct_groups[tipsy_analyzer.nMinMembers])


def test_singlecore_fof(tipsy_analyzer):
    """Check that singlecore cython-wrapped code works as expected"""
    from spark_fof.fof import fof
    ps = np.concatenate(tipsy_analyzer.particle_rdd.collect())
    n_groups = fof.run(ps, tau, tipsy_analyzer.nMinMembers)
