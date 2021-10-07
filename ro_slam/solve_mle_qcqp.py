import numpy as np
from os.path import expanduser, join
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt  # type: ignore
import scipy.linalg as la  # type: ignore
from pydrake.solvers.mathematicalprogram import MathematicalProgram, Solve  # type: ignore
from pydrake.solvers.ipopt import IpoptSolver
from pydrake.solvers.snopt import SnoptSolver

from ro_slam.qcqp_utils import (
    pin_first_pose,
    add_pose_variables,
    add_landmark_variables,
    add_distance_variables,
    add_distances_cost,
    add_odom_cost,
    set_rotation_init_gt,
    set_rotation_init_compose,
    set_translation_init_gt,
    set_translation_init_compose,
    set_distance_init_gt,
    set_landmark_init_gt,
)
from ro_slam.factor_graph.parse_factor_graph import parse_factor_graph_file
from ro_slam.factor_graph.factor_graph import FactorGraphData
from ro_slam.utils import get_theta_from_matrix, _check_rotation_matrix


def print_state(
    result,
    translations: List[np.ndarray],
    rotations: List[np.ndarray],
    idx: int,
):
    """
    Prints the current state of the result

    Args:
        result (MathematicalProgram): the result of the solution
        translations (List[np.ndarray]): the translations
        rotations (List[np.ndarray]): the rotations
        idx (int): the index of the pose to print
    """
    trans_solve = result.GetSolution(translations[idx]).round(decimals=2)
    rot_solve = result.GetSolution(rotations[idx])
    theta_solve = get_theta_from_matrix(rot_solve)

    trans_string = np.array2string(trans_solve, precision=1, floatmode="fixed")

    status = (
        f"State {idx}"
        + f" | Translation: {trans_string}"
        + f" | Rotation: {theta_solve:.2f}"
    )
    print(status)


def save_results_to_file(
    result, filepath: str, translations: List[np.ndarray], rotations: List[np.ndarray]
):
    """
    Saves the results to a file

    args:
        result (MathematicalProgram): the result of the solution
        filepath (str): the path to save the results to
        translations (List[np.ndarray]): the translations
        rotations (List[np.ndarray]): the rotations
    """
    with open(filepath, "w") as f:
        for idx in range(len(translations)):
            trans_solve = result.GetSolution(translations[idx]).round(decimals=2)
            rot_solve = result.GetSolution(rotations[idx])
            theta_solve = get_theta_from_matrix(rot_solve)

            trans_string = np.array2string(trans_solve, precision=1, floatmode="fixed")
            status = (
                f"State {idx}"
                + f" | Translation: {trans_string}"
                + f" | Rotation: {theta_solve:.2f}\n"
            )
            f.write(status)

        f.write(f"Is optimization successful? {result.is_success()}\n")
        f.write(f"optimal cost: {result.get_optimal_cost()}")

    print(f"Results saved to: {filepath}\n")


def check_rotations(result, rotations):
    """
    checks that results are valid rotations
    """
    for rotation in rotations:
        rot_result = result.GetSolution(rotation)
        _check_rotation_matrix(rot_result)


def solve_mle_problem(
    data: FactorGraphData,
    solver: str,
    verbose: bool,
    save_results: bool,
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
    """
    solver_options = ["ipopt", "snopt", "default"]
    assert solver in solver_options, f"Invalid solver, must be from: {solver_options}"

    model = MathematicalProgram()

    # form objective function
    translations, rotations = add_pose_variables(model, data)
    assert len(translations) == len(rotations)

    landmarks = add_landmark_variables(model, data)
    distances = add_distance_variables(model, data, translations, landmarks)

    add_distances_cost(model, distances, data)
    add_odom_cost(model, translations, rotations, data)

    # pin first pose at origin
    pin_first_pose(model, translations[0], rotations[0])

    ### Rotation Initialization
    set_rotation_init_gt(model, rotations, data)
    # set_rotation_init_compose(model, rotations, data)

    ### Translation Initialization
    set_translation_init_gt(model, translations, data)
    # set_translation_init_compose(model, translations, data)

    ### Distance Initialization
    set_distance_init_gt(model, distances, data)

    ### Landmark Initialization
    set_landmark_init_gt(model, landmarks, data)

    # perform optimization
    print("Solving MLE problem...")

    if solver == "default":
        result = Solve(model)
    elif solver == "ipopt":
        ipopt_solver = IpoptSolver()
        # model.SetSolverOption(ipopt_solver.solver_id(), "print_level", 5)
        result = ipopt_solver.Solve(model)
    elif solver == "snopt":
        snopt_solver = SnoptSolver()
        # debug_path = expanduser("~/snopt_debug.out")
        # model.SetSolverOption(snopt_solver.solver_id(), "print file", debug_path)
        result = snopt_solver.Solve(model)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    check_rotations(result, rotations)

    if verbose:
        for i in range(len(translations)):
            print_state(result, translations, rotations, i)

        print(f"Is optimization successful? {result.is_success()}")
        print(f"optimal cost: {result.get_optimal_cost()}")

    if save_results:
        save_results_to_file(result, results_filepath, translations, rotations)
