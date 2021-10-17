import numpy as np
from typing import List, Tuple, Dict, Union
import attr
import pickle
import re
import time

from pydrake.solvers.mathematicalprogram import MathematicalProgram  # type: ignore
from pydrake.solvers.ipopt import IpoptSolver
from pydrake.solvers.snopt import SnoptSolver
from pydrake.solvers.gurobi import GurobiSolver
from pydrake.solvers.mosek import MosekSolver
from factor_graph.factor_graph import FactorGraphData

from ro_slam.utils.qcqp_utils import (
    pin_first_pose,
    add_pose_variables,
    add_landmark_variables,
    add_distance_variables,
    add_distances_cost,
    add_odom_cost,
    set_rotation_init_gt,
    set_rotation_init_compose,
    set_rotation_init_random_rotation,
    set_translation_init_gt,
    set_translation_init_compose,
    set_translation_init_random,
    set_distance_init_gt,
    set_distance_init_measured,
    set_distance_init_random,
    set_landmark_init_gt,
    set_landmark_init_random,
)
from ro_slam.utils.eval_utils import (
    get_solved_values,
    print_state,
    save_results_to_file,
)
from ro_slam.utils.plot_utils import (
    plot_error,
)


@attr.s(frozen=True)
class SolverParams:
    solver: str = attr.ib()
    verbose: bool = attr.ib()
    save_results: bool = attr.ib()
    use_socp_relax: bool = attr.ib()
    use_orthogonal_constraint: bool = attr.ib()
    init_technique: str = attr.ib()


def get_solver(
    solver_name: str,
) -> Union[IpoptSolver, SnoptSolver, GurobiSolver, MosekSolver]:
    """
    Returns the solver for the given name
    """
    if solver_name == "ipopt":
        return IpoptSolver()
    elif solver_name == "snopt":
        return SnoptSolver()
    elif solver_name == "gurobi":
        return GurobiSolver()
    elif solver_name == "mosek":
        return MosekSolver()
    else:
        raise ValueError(f"Unknown solver: {solver_name}")


def solve_mle_problem(
    data: FactorGraphData,
    solver_params: SolverParams,
    results_filepath: str,
):
    """
    Takes the data describing the problem and returns the MLE solution to the
    poses and landmark positions

    args:
        data (FactorGraphData): the data describing the problem
        solver (str): the solver to use [ipopt, snopt, default]
        verbose (bool): whether to show verbose solver output
        save_results (bool): whether to save the results to a file
        results_filepath (str): the path to save the results to
        use_socp_relax (bool): whether to use socp relaxation on distance
            variables
        use_orthogonal_constraint (bool): whether to use orthogonal
            constraint on rotation variables
    """
    solver_options = ["mosek", "gurobi", "ipopt", "snopt", "default"]
    assert (
        solver_params.solver in solver_options
    ), f"Invalid solver, must be from: {solver_options}"

    init_options = ["gt", "compose", "random"]
    assert (
        solver_params.init_technique in init_options
    ), f"Invalid init_technique, must be from: {init_options}"

    if solver_params.solver in ["mosek", "gurobi"]:
        assert (
            solver_params.use_socp_relax and not solver_params.use_orthogonal_constraint
        ), "Mosek and Gurobi solver only used to solve convex problems"

    model = MathematicalProgram()

    # form objective function
    translations, rotations = add_pose_variables(
        model, data, solver_params.use_orthogonal_constraint
    )
    print("Added pose variables")
    assert (translations.keys()) == (rotations.keys())

    landmarks = add_landmark_variables(model, data)
    print("Added landmark variables")
    distances = add_distance_variables(
        model, data, translations, landmarks, solver_params.use_socp_relax
    )
    print("Added distance variables")

    add_distances_cost(model, distances, data)
    add_odom_cost(model, translations, rotations, data)

    # pin first pose at origin
    pin_first_pose(model, translations["A0"], rotations["A0"])

    if solver_params.init_technique == "gt":
        set_rotation_init_gt(model, rotations, data)
        set_translation_init_gt(model, translations, data)
        set_distance_init_gt(model, distances, data)
        set_landmark_init_gt(model, landmarks, data)
    elif solver_params.init_technique == "compose":
        set_rotation_init_compose(model, rotations, data)
        set_translation_init_compose(model, translations, data)
        set_distance_init_measured(model, distances, data)
        set_landmark_init_random(model, landmarks, data)
    elif solver_params.init_technique == "random":
        set_rotation_init_random_rotation(model, rotations)
        set_translation_init_random(model, translations)
        set_distance_init_random(model, distances)
        set_landmark_init_random(model, landmarks)

    # perform optimization
    print("Solving MLE problem...")

    t_start = time.time()
    try:
        solver = get_solver(solver_params.solver)
        result = solver.Solve(model)
    except Exception as e:
        print("Error: ", e)
        return
    t_end = time.time()
    print(f"Solved in {t_end - t_start} seconds")

    # check_rotations(result, rotations)

    solved_translations, solved_rotations, solved_landmarks = get_solved_values(
        result, translations, rotations, landmarks
    )
    solved_vals = {
        "translations": solved_translations,
        "rotations": solved_rotations,
        "landmarks": solved_landmarks,
    }

    if solver_params.verbose:
        for pose_key in translations.keys():
            print_state(result, translations, rotations, pose_key)

        print(f"Is optimization successful? {result.is_success()}")
        print(f"optimal cost: {result.get_optimal_cost()}")

    if solver_params.save_results:
        save_results_to_file(result, solved_vals, results_filepath)

    grid_size_str = re.search(r"\d+_grid", results_filepath).group(0)  # type: ignore
    grid_size = int(grid_size_str.split("_")[0])
    plot_error(data, solved_vals, grid_size)
