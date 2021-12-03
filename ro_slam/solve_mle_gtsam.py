import re
import time

from gtsam.gtsam import (
    NonlinearFactorGraph,
    ISAM2Params,
    ISAM2DoglegParams,
    ISAM2,
    Values,
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
from ro_slam.utils.matrix_utils import make_transformation_matrix

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
    init_options = ["gt", "compose", "random", "custom"]
    assert (
        solver_params.init_technique in init_options
    ), f"Invalid init_technique, must be from: {init_options}"

    factor_graph = NonlinearFactorGraph()
    initial_values = Values()

    # form objective function
    gt_ut.add_distances_cost(factor_graph, data)
    gt_ut.add_odom_cost(factor_graph, data)
    gt_ut.add_loop_closure_cost(factor_graph, data)

    # pin first pose at origin
    gt_ut.pin_first_pose(factor_graph, data)

    # choose an initialization strategy
    if solver_params.init_technique == "gt":
        gt_ut.set_pose_init_gt(initial_values, data)
        gt_ut.set_landmark_init_gt(initial_values, data)
    elif solver_params.init_technique == "compose":
        gt_ut.set_pose_init_compose(initial_values, data)
        gt_ut.set_landmark_init_random(initial_values, data)
    elif solver_params.init_technique == "random":
        gt_ut.set_pose_init_random(initial_values, data)
        gt_ut.set_landmark_init_random(initial_values, data)
    elif solver_params.init_technique == "custom":
        assert (
            solver_params.custom_init_file is not None
        ), "Must provide custom_init_filepath if using custom init"
        custom_vals = load_custom_init_file(solver_params.custom_init_file)
        init_rotations = custom_vals.rotations
        init_translations = custom_vals.translations
        init_poses = {
            key: make_transformation_matrix(init_rotations[key], init_translations[key])
            for key in init_rotations.keys()
        }
        init_landmarks = custom_vals.landmarks
        gt_ut.set_pose_init_custom(initial_values, init_poses)
        gt_ut.set_landmark_init_custom(initial_values, init_landmarks)

    # perform optimization
    print("Initializing solver...")

    # initialize the ISAM2 instance
    parameters = ISAM2Params()
    parameters.setOptimizationParams(ISAM2DoglegParams())
    print(f"ISAM Params: {parameters}")
    isam_solver = ISAM2(parameters)
    isam_solver.update(factor_graph, initial_values)

    # run the optimization
    print("Solving ...")
    t_start = time.time()
    try:
        if solver_params.verbose:
            print("Do not have verbose settings for GTSAM yet")

        gtsam_result = isam_solver.calculateEstimate()
    except Exception as e:
        print("Error: ", e)
        return
    t_end = time.time()
    tot_time = t_end - t_start
    print(f"Solved in {tot_time} seconds")

    # get the results and save if desired
    solution_vals = gt_ut.get_solved_values(gtsam_result, tot_time, data)
    if solver_params.save_results:
        save_results_to_file(
            solution_vals,
            solved_cost=factor_graph.error(gtsam_result),
            solve_success=True,
            filepath=results_filepath,
        )

    # get the grid size to aid in plotting
    grid_size_search = re.search(r"\d+_grid", results_filepath)
    assert grid_size_search is not None, "Grid size not found in results filepath"
    grid_size = int(grid_size_search.group(0).split("_")[0])

    print(solution_vals.distances)

    # perform plotting
    if solver_params.init_technique == "custom":
        plot_error_with_custom_init(data, solution_vals, custom_vals, grid_size)
    else:
        # do not use custom init so we just compare to GT pose
        plot_error(data, solution_vals, grid_size)


if __name__ == "__main__":
    import argparse
    from os.path import join
    from py_factor_graph.parse_factor_graph import (
        parse_efg_file,
        parse_pickle_file,
    )

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "data_dir", type=str, help="Path to the directory the PyFactorGraph is held in"
    )
    arg_parser.add_argument("pyfg_filename", type=str, help="name of the PyFactorGraph")
    arg_parser.add_argument(
        "results_dir", type=str, help="Path to the directory the results are saved to"
    )
    arg_parser.add_argument(
        "results_filename", type=str, help="name of the results file"
    )
    arg_parser.add_argument(
        "init_technique", type=str, help="Initialization technique to use"
    )
    arg_parser.add_argument(
        "custom_init_file",
        type=str,
        help="Path to the custom initialization file",
        default=None,
        required=False,
    )
    args = arg_parser.parse_args()

    solver_params = GtsamSolverParams(
        verbose=True,
        save_results=True,
        init_technique=args.init_technique,
        custom_init_file=args.custom_init_file,
    )

    fg_filepath = join(args.data_dir, args.pyfg_filename)
    if fg_filepath.endswith(".pickle"):
        fg = parse_pickle_file(fg_filepath)
    elif fg_filepath.endswith(".fg"):
        fg = parse_efg_file(fg_filepath)
    else:
        raise ValueError(f"Unknown file type: {fg_filepath}")
    print(f"Loaded data: {fg_filepath}")

    results_filepath = join(args.results_dir, args.results_filename)
    solve_mle_gtsam(fg, solver_params, results_filepath)
