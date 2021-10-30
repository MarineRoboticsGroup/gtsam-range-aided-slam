import re
import time

from gtsam.gtsam import (
    NonlinearFactorGraph,
    ISAM2Params,
    ISAM2DoglegParams,
    ISAM2,
)

from py_factor_graph.factor_graph import FactorGraphData

from ro_slam.utils.plot_utils import (
    plot_error,
    plot_error_with_custom_init,
)
from ro_slam.utils.solver_utils import (
    GtsamSolverParams,
    SolverResults,
    save_results_to_file,
    load_custom_init_file,
)

import ro_slam.utils.gtsam_utils as gt_ut


def solve_mle_gtsam(
    data: FactorGraphData,
    solver_params: GtsamSolverParams,
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
    init_options = ["gt", "compose", "random", "none", "custom"]
    assert (
        solver_params.init_technique in init_options
    ), f"Invalid init_technique, must be from: {init_options}"

    factor_graph = NonlinearFactorGraph()

    # form objective function
    gt_ut.add_distances_cost(factor_graph, data)
    gt_ut.add_odom_cost(factor_graph, data)
    gt_ut.add_loop_closure_cost(factor_graph, data)

    # pin first pose at origin
    gt_ut.pin_first_pose(factor_graph, data)

    # choose an initialization strategy
    if solver_params.init_technique == "gt":
        gt_ut.set_pose_init_gt(factor_graph, data)
        gt_ut.set_landmark_init_gt(factor_graph, data)
    elif solver_params.init_technique == "compose":
        gt_ut.set_pose_init_compose(factor_graph, data)
        gt_ut.set_landmark_init_random(factor_graph, data)
    elif solver_params.init_technique == "random":
        gt_ut.set_pose_init_random(factor_graph, data)
        gt_ut.set_landmark_init_random(factor_graph, data)
    elif solver_params.init_technique == "custom":
        assert (
            solver_params.custom_init_file is not None
        ), "Must provide custom_init_filepath if using custom init"
        custom_vals = load_custom_init_file(solver_params.custom_init_file)
        init_rotations = custom_vals.rotations
        init_translations = custom_vals.translations
        init_landmarks = custom_vals.landmarks
        gt_ut.set_pose_init_custom(factor_graph, init_rotations)
        gt_ut.set_landmark_init_custom(factor_graph, init_landmarks)

    # perform optimization
    print("Starting solver...")

    t_start = time.time()
    try:
        raise NotImplementedError("Solver not implemented")
        # solver = get_drake_solver(solver_params.solver)
        # if solver_params.verbose:
        #     set_drake_solver_verbose(factor_graph, solver)

        # result = solver.Solve(factor_graph)
    except Exception as e:
        print("Error: ", e)
        return
    t_end = time.time()
    tot_time = t_end - t_start
    print(f"Solved in {tot_time} seconds")

    # TODO get success measure (does GTSAM offer this?)
    # print(f"Solver success: {result.is_success()}")

    # check_rotations(result, rotations)

    # TODO get solution
    # solution_vals = gt_ut.get_solved_values(
    #     result, tot_time, translations, rotations, landmarks, distances
    # )

    # TODO finish saving
    # if solver_params.save_results:
    #     save_results_to_file(result, solution_vals, results_filepath)

    grid_size_str = re.search(r"\d+_grid", results_filepath).group(0)  # type: ignore
    grid_size = int(grid_size_str.split("_")[0])

    # TODO finish plotting
    # if solver_params.init_technique == "custom":
    #     plot_error_with_custom_init(data, solution_vals, custom_vals, grid_size)
    # else:
    #     # do not use custom init so we just compare to GT pose
    #     plot_error(data, solution_vals, grid_size)
