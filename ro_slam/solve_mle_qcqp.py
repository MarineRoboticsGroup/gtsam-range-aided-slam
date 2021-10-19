from typing import Optional
import attr
import re
import time

from pydrake.solvers.mathematicalprogram import MathematicalProgram  # type: ignore
from factor_graph.factor_graph import FactorGraphData

from ro_slam.utils.qcqp_utils import (
    pin_first_pose,
    add_pose_variables,
    add_landmark_variables,
    add_distance_variables,
    add_distances_cost,
    add_odom_cost,
    add_loop_closure_cost,
    set_rotation_init_gt,
    set_rotation_init_compose,
    set_rotation_init_random_rotation,
    set_rotation_init_custom,
    set_translation_init_gt,
    set_translation_init_compose,
    set_translation_init_random,
    set_translation_init_custom,
    set_distance_init_gt,
    set_distance_init_measured,
    set_distance_init_random,
    set_distance_init_custom,
    set_landmark_init_gt,
    set_landmark_init_random,
    set_landmark_init_custom,
)
from ro_slam.utils.plot_utils import (
    plot_error,
    plot_error_with_custom_init,
)
from ro_slam.utils.solver_utils import (
    SolverParams,
    SolverResults,
    get_solved_values,
    save_results_to_file,
    load_custom_init_file,
    get_solver,
    set_solver_verbose,
)


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

    init_options = ["gt", "compose", "random", "none"]
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
    add_loop_closure_cost(model, translations, rotations, data)

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
    elif solver_params.init_technique == "custom":
        custom_vals = load_custom_init_file(solver_params.custom_init_file)
        init_rotations = custom_vals["rotations"]
        init_translations = custom_vals["translations"]
        init_landmarks = custom_vals["landmarks"]
        init_distances = custom_vals["distances"]
        set_rotation_init_custom(model, rotations, init_rotations)
        set_translation_init_custom(model, translations, init_translations)
        set_landmark_init_custom(model, landmarks, init_landmarks)
        set_distance_init_custom(model, distances, init_distances)

    # perform optimization
    print("Starting solver...")

    t_start = time.time()
    try:
        solver = get_solver(solver_params.solver)
        if solver_params.verbose:
            set_solver_verbose(model, solver)

        result = solver.Solve(model)
    except Exception as e:
        print("Error: ", e)
        return
    t_end = time.time()
    print(f"Solved in {t_end - t_start} seconds")

    # check_rotations(result, rotations)

    solution_vals = get_solved_values(
        result, translations, rotations, landmarks, distances
    )

    if solver_params.save_results:
        save_results_to_file(result, solution_vals, results_filepath)

    grid_size_str = re.search(r"\d+_grid", results_filepath).group(0)  # type: ignore
    grid_size = int(grid_size_str.split("_")[0])

    if solver_params.init_technique == "custom":
        plot_error_with_custom_init(data, solution_vals, custom_vals, grid_size)
    else:
        # do not solve local so only print the relaxed solution
        plot_error(data, solution_vals, grid_size)
