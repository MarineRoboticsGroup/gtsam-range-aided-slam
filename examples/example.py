import os
from os.path import join, expanduser
import sys
from typing import List

sys.path.insert(0, os.path.abspath(".."))

from ro_slam.factor_graph.parse_factor_graph import parse_factor_graph_file
from ro_slam.solve_mle_qcqp import solve_mle_problem


def get_folders_in_dir(path) -> List[str]:
    return [join(path, f) for f in os.listdir(path) if os.path.isdir(join(path, f))]


def get_files_in_dir(path) -> List[str]:
    return [join(path, f) for f in os.listdir(path) if os.path.isfile(join(path, f))]


def get_factor_graph_file_in_dir(path) -> str:
    fg_files = [x for x in get_files_in_dir(path) if x.endswith(".fg")]
    assert (
        len(fg_files) == 1
    ), "There should be only one factor graph file in the directory"
    return fg_files[0]


if __name__ == "__main__":
    base_dir = expanduser(join("~", "qcqp-range-only-slam", "data"))
    data_folders = get_folders_in_dir(base_dir)
    data_folders.sort(key=lambda x: len(x))

    solver = "snopt"
    verbose = False
    save_results = True

    for d_fold in data_folders:
        experiment_folders = get_folders_in_dir(d_fold)
        experiment_folders.sort()
        for exp_fold in experiment_folders:
            fg_filepath = get_factor_graph_file_in_dir(exp_fold)
            results_filepath = join(exp_fold, "results.txt")
            print(fg_filepath)
            fg = parse_factor_graph_file(fg_filepath)
            solve_mle_problem(fg, solver, verbose, save_results, results_filepath)
        print()

    # filepath = (
    #     "/home/alan/data/example_factor_graphs/15_loop_clos/test_9/factor_graph.fg"
    # )
