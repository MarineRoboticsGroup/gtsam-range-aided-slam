import os
from os.path import join, expanduser, abspath, realpath
import sys

file_dir = os.path.dirname(os.path.realpath(__file__))
ro_slam_dir = abspath(realpath(join(file_dir, "..")))
sys.path.insert(0, ro_slam_dir)


from py_factor_graph.parsing import (
    parse_efg_file,
    parse_pickle_file,
)
from py_factor_graph.factor_graph import FactorGraphData
from ro_slam.utils.solver_utils import GtsamSolverParams
from ro_slam.solve_mle_gtsam import solve_mle_gtsam

from example_utils import recursively_find_pickle_files, get_gtsam_results_filename

if __name__ == "__main__":
    # base_dir = expanduser(join("~", "data", "manhattan"))
    base_dir = expanduser(join("~", "final_pkgs", "py-rosbag-parsing", "py_rosbag_parser"))
    solver_params = GtsamSolverParams(
        verbose=True,
        save_results=True,
        init_technique="gt",
        custom_init_file=None,
    )
    # results_filetype = "pickle"
    results_filetype = "tum"

    # do a recursive search and then test on all of the .pickle files found
    pickle_files = recursively_find_pickle_files(base_dir)

    assert len(pickle_files) > 0, "No pickle files found in {}".format(base_dir)
    for pickle_dir, pickle_file in pickle_files:

        # if not pickle_file == "factor_graph.pickle":
        #     continue

        # get the factor graph filepath
        fg_filepath = join(pickle_dir, pickle_file)

        # get the file name to save results to
        results_file_name = get_gtsam_results_filename(solver_params, results_filetype)
        results_filepath = join(pickle_dir, results_file_name)

        # if "100_timesteps" not in fg_filepath:
        #     continue
        # if "3_beacons" not in fg_filepath:
        #     continue
        # if "50_loop"not  in fg_filepath:
        #     continue
        # if "100_loop"not  in fg_filepath:
        #     continue

        if fg_filepath.endswith(".pickle"):
            fg = parse_pickle_file(fg_filepath)
        elif fg_filepath.endswith(".fg"):
            fg = parse_efg_file(fg_filepath)
        else:
            raise ValueError(f"Unknown file type: {fg_filepath}")
        print(f"Loaded data: {fg_filepath}")
        print(fg.odom_measurements[1][955:965])
        print(fg.range_measurements)
        solve_mle_gtsam(fg, solver_params, results_filepath)

    print()
