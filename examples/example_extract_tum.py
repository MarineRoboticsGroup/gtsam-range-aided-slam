import os
from os.path import join, expanduser, abspath, realpath
import sys
import pickle

file_dir = os.path.dirname(os.path.realpath(__file__))
ro_slam_dir = abspath(realpath(join(file_dir, "..")))
sys.path.insert(0, ro_slam_dir)

import ro_slam

if __name__ == "__main__":
    file_location = "/home/tim/final_ws/other/gurobi_initnone_socp_noorth_results.pickle"
    result = pickle.load(open(file_location, 'rb'))
    # print(type(result))

    ro_slam.utils.solver_utils.save_results_to_file(result, 0, True, "traj.tum")