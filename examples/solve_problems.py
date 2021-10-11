import os
from os.path import join, expanduser
import sys
import re
from typing import List, Tuple

sys.path.insert(0, os.path.abspath(".."))

from ro_slam.factor_graph.parse_factor_graph import parse_efg_factor_graph_file, parse_pickle_factor_graph_file
from ro_slam.solve_mle_qcqp import solve_mle_problem, SolverParams


def get_folders_in_dir(path) -> List[str]:
    return [join(path, f) for f in os.listdir(path) if os.path.isdir(join(path, f))]


def get_files_in_dir(path) -> List[str]:
    return [join(path, f) for f in os.listdir(path) if os.path.isfile(join(path, f))]

def recursively_find_pkl_files(dir) -> List[Tuple[str, str]]:
    """Recursively finds all .pkl files in the directory and its subdirectories

    Args:
        dir (str): the directory to search in

    Returns:
        List[Tuple[str, str]]: a list of tuples of the form (root, file_name)

    """

    def num_timesteps_from_path(path: str) -> int:
        trailing_phrase = "_timesteps"
        info = re.search(r"\d+"+trailing_phrase, path).group(0) # type: ignore
        num_timesteps = int(info[:-len(trailing_phrase)])
        return num_timesteps

    pkl_files = []
    for root, _, files in os.walk(dir):
        for file in files:
            if file.endswith(".pkl"):
                pkl_files.append((root, file))

    pkl_files.sort(key=lambda x: num_timesteps_from_path(x[0]))
    return pkl_files


def get_factor_graph_file_in_dir(path) -> str:

    # prefer pickle files but also check for .fg files
    pkl_files = [x for x in get_files_in_dir(path) if x.endswith(".pkl")]
    if len(pkl_files) >= 1:
        assert len(pkl_files) == 1, "There should be only one pkl file in the directory"
        return pkl_files[0]

    efg_files = [x for x in get_files_in_dir(path) if x.endswith(".fg")]
    if len(efg_files) >= 1:
        assert (
            len(efg_files) == 1
        ), "There should be only one factor graph file in the directory"
        return efg_files[0]

    raise ValueError(f"No factor graph file found in the directory: {path}")


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
    solver_params = SolverParams(
        solver="gurobi",
        verbose=False,
        save_results=True,
        use_socp_relax=True,
        use_orthogonal_constraint=False,
    )

    # do a recursive search and then test on all of the .pkl files found
    pkl_files = recursively_find_pkl_files(base_dir)
    for pkl_dir, pkl_file in pkl_files:

        # get the factor graph filepath
        fg_filepath = join(pkl_dir, pkl_file)

        # get the file name to save results to
        results_file_name = get_results_filename(
            solver_params.solver,
            solver_params.use_socp_relax,
            solver_params.use_orthogonal_constraint,
        )
        results_filepath = join(pkl_dir, results_file_name)

        print(fg_filepath)
        if fg_filepath.endswith(".pkl"):
            fg = parse_pickle_factor_graph_file(fg_filepath)
        elif fg_filepath.endswith(".fg"):
            fg = parse_efg_factor_graph_file(fg_filepath)
        else:
            raise ValueError(f"Unknown file type: {fg_filepath}")

        solve_mle_problem(fg, solver_params, results_filepath)
    print()

# filepath = (
#     "/home/alan/data/example_factor_graphs/15_loop_clos/test_9/factor_graph.fg"
# )
