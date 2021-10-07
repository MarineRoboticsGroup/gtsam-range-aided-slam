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


def get_results_filename(
    solver: str, use_socp_relax: bool, use_orthogonal_constraints: bool
) -> str:
    """Returns the name of the results file

    Args:
        solver (str): The solver used to solve the problem
        use_socp_relax (bool): whether the problem is solved with SOCP relaxation
        use_orthogonal_constraints (bool): whether the problem is solved with
            orthogonal constraints

    Returns:
        str: the file name giving details of the solver params
    """
    file_name = f"{solver}_"

    # add in indicator for SOCP relaxation
    if use_socp_relax:
        file_name += "socp"
    else:
        file_name += "nosocp"
    file_name += "_"

    # add in indicator for orthogonal constraints
    if use_orthogonal_constraints:
        file_name += "orth"
    else:
        file_name += "noorth"
    file_name += "_"

    # add in results.txt and return
    file_name += "results.txt"
    return file_name


if __name__ == "__main__":
    base_dir = expanduser(join("~", "qcqp-range-only-slam", "data"))
    data_folders = get_folders_in_dir(base_dir)
    data_folders.sort(key=lambda x: len(x))

    solver = "mosek"
    verbose = False
    save_results = True
    socp_relaxation = True
    orthogonal_constraints = False

    # get all of the data folders (broken down by different simulation settings)
    for d_fold in data_folders:

        # temporarily skipping the larger problems
        if "1000" in d_fold or "100" in d_fold:
            continue

        # get the folders holding the individual experiments just with different
        # random seeds
        experiment_folders = get_folders_in_dir(d_fold)
        experiment_folders.sort()
        for exp_fold in experiment_folders:

            # get the factor graph file
            fg_filepath = get_factor_graph_file_in_dir(exp_fold)

            # get the file name to save results to
            results_file_name = get_results_filename(
                solver, socp_relaxation, orthogonal_constraints
            )
            results_filepath = join(exp_fold, results_file_name)

            print(fg_filepath)
            fg = parse_factor_graph_file(fg_filepath)
            solve_mle_problem(
                fg,
                solver,
                verbose,
                save_results,
                results_filepath,
                socp_relaxation,
                orthogonal_constraints,
            )
        print()

    # filepath = (
    #     "/home/alan/data/example_factor_graphs/15_loop_clos/test_9/factor_graph.fg"
    # )
