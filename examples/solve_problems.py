import os
from os.path import join, expanduser
import sys
import re
from typing import List, Tuple

sys.path.insert(0, os.path.abspath(".."))

from factor_graph.parse_factor_graph import (
    parse_efg_file,
    parse_pickle_file,
)
from ro_slam.solve_mle_qcqp import solve_mle_problem, SolverParams


def get_folders_in_dir(path) -> List[str]:
    return [join(path, f) for f in os.listdir(path) if os.path.isdir(join(path, f))]


def get_files_in_dir(path) -> List[str]:
    return [join(path, f) for f in os.listdir(path) if os.path.isfile(join(path, f))]


def recursively_find_pickle_files(dir) -> List[Tuple[str, str]]:
    """Recursively finds all .pickle files in the directory and its subdirectories

    Args:
        dir (str): the directory to search in

    Returns:
        List[Tuple[str, str]]: a list of tuples of the form (root, file_name)

    """

    def num_timesteps_from_path(path: str) -> int:
        trailing_phrase = "_timesteps"
        info = re.search(r"\d+" + trailing_phrase, path).group(0)  # type: ignore
        num_timesteps = int(info[: -len(trailing_phrase)])
        return num_timesteps

    pickle_files = []
    for root, _, files in os.walk(dir):
        for file in files:
            if file.endswith(".pickle"):
                pickle_files.append((root, file))

    pickle_files.sort(key=lambda x: num_timesteps_from_path(x[0]))
    return pickle_files


def get_factor_graph_file_in_dir(path) -> str:

    # prefer pickle files but also check for .fg files
    pickle_files = [x for x in get_files_in_dir(path) if x.endswith(".pickle")]
    if len(pickle_files) >= 1:
        assert (
            len(pickle_files) == 1
        ), "There should be only one pickle file in the directory"
        return pickle_files[0]

    efg_files = [x for x in get_files_in_dir(path) if x.endswith(".fg")]
    if len(efg_files) >= 1:
        assert (
            len(efg_files) == 1
        ), "There should be only one factor graph file in the directory"
        return efg_files[0]

    raise ValueError(f"No factor graph file found in the directory: {path}")


def get_results_filename(
    solver_params: SolverParams,
) -> str:
    """Returns the name of the results file

    Args:
        solver_params (SolverParams): the solver parameters

    Returns:
        str: the file name giving details of the solver params
    """
    file_name = f"{solver_params.solver}_"

    file_name += f"init{solver_params.init_technique}_"

    # add in indicator for SOCP relaxation
    if solver_params.use_socp_relax:
        file_name += "socp"
    else:
        file_name += "nosocp"
    file_name += "_"

    # add in indicator for orthogonal constraints
    if solver_params.use_orthogonal_constraint:
        file_name += "orth"
    else:
        file_name += "noorth"
    file_name += "_"

    # add in results.txt and return
    file_name += "results.txt"
    return file_name


if __name__ == "__main__":
    base_dir = expanduser(join("~", "data", "example_factor_graphs"))
    solver_params = SolverParams(
        solver="gurobi",
        verbose=False,
        save_results=True,
        use_socp_relax=True,
        use_orthogonal_constraint=False,
        init_technique="random",
    )

    # do a recursive search and then test on all of the .pickle files found
    pickle_files = recursively_find_pickle_files(base_dir)
    for pickle_dir, pickle_file in pickle_files:

        # get the factor graph filepath
        fg_filepath = join(pickle_dir, pickle_file)

        # get the file name to save results to
        results_file_name = get_results_filename(solver_params)
        results_filepath = join(pickle_dir, results_file_name)

        print(fg_filepath)
        if "100_timesteps" not in fg_filepath:
            continue
        if "1_beacons" not in fg_filepath:
            continue

        if fg_filepath.endswith(".pickle"):
            fg = parse_pickle_file(fg_filepath)
        elif fg_filepath.endswith(".fg"):
            fg = parse_efg_file(fg_filepath)
        else:
            raise ValueError(f"Unknown file type: {fg_filepath}")
        print(f"Loaded data: {fg_filepath}")

        solve_mle_problem(fg, solver_params, results_filepath)
    print()
