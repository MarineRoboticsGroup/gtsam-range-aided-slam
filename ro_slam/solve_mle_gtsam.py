import re
import time

import logging, coloredlogs

logger = logging.getLogger(__name__)
field_styles = {
    "filename": {"color": "green"},
    "filename": {"color": "green"},
    "levelname": {"bold": True, "color": "black"},
    "name": {"color": "blue"},
}
coloredlogs.install(
    level="INFO",
    fmt="[%(filename)s:%(lineno)d] %(name)s %(levelname)s - %(message)s",
    field_styles=field_styles,
)

from gtsam.gtsam import (
    NonlinearFactorGraph,
    ISAM2Params,
    ISAM2DoglegParams,
    ISAM2,
    Values,
)

from py_factor_graph.factor_graph import FactorGraphData
from py_factor_graph.utils.solver_utils import (
    SolverResults,
    save_results_to_file,
    load_custom_init_file,
)

from ro_slam.utils.plot_utils import plot_error
from ro_slam.utils.gtsam_utils import GtsamSolverParams
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
    logger.debug(f"Running GTSAM solver with {solver_params}")

    unconnected_variables = data.unconnected_variable_names
    assert (
        len(unconnected_variables) == 0
    ), f"Found {sorted(list(unconnected_variables))} unconnected variables. "

    factor_graph = NonlinearFactorGraph()
    gt_ut.add_all_costs(factor_graph, data)
    initial_values = gt_ut.get_initial_values(solver_params)

    # pin first pose at origin
    gt_ut.pin_first_pose(factor_graph, data)
    # gt_ut.pin_first_landmark(factor_graph, data)

    # perform optimization
    logger.debug("Initializing solver...")

    # initialize the ISAM2 instance
    parameters = ISAM2Params()
    parameters.setOptimizationParams(ISAM2DoglegParams())
    logger.debug(f"ISAM Params: {parameters}")
    isam_solver = ISAM2(parameters)
    isam_solver.update(factor_graph, initial_values)

    # run the optimization
    logger.debug("Solving ...")
    t_start = time.time()
    try:
        gtsam_result = isam_solver.calculateEstimate()
    except Exception as e:
        logger.error("Error: ", e)
        return
    t_end = time.time()
    tot_time = t_end - t_start
    logger.debug(f"Solved in {tot_time} seconds")

    # get the cost at the solution
    cost = factor_graph.error(gtsam_result)

    # get the cost for each factor
    # gt_ut.generate_detailed_report_of_factor_costs(
    #     factor_graph, gtsam_result
    # )

    # print cost in scientific notation
    logger.debug(f"Cost at solution: {cost:.2e}")
    # logger.info(f"Cost at solution: {cost}")

    # get the results and save if desired
    solution_vals = gt_ut.get_solved_values(gtsam_result, tot_time, data, cost)
    if solver_params.save_results:
        save_results_to_file(
            solution_vals,
            solved_cost=factor_graph.error(gtsam_result),
            solve_success=True,
            filepath=results_filepath,
        )

    return solution_vals


if __name__ == "__main__":
    import argparse
    from os.path import join, isfile
    from py_factor_graph.parsing.parse_efg_file import parse_efg_file
    from py_factor_graph.parsing.parse_pickle_file import parse_pickle_file

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
    )
    args = arg_parser.parse_args()

    fg_filepath = join(args.data_dir, args.pyfg_filename)
    if fg_filepath.endswith(".pickle") or fg_filepath.endswith(".pkl"):
        fg = parse_pickle_file(fg_filepath)
    elif fg_filepath.endswith(".fg"):
        fg = parse_efg_file(fg_filepath)
    else:
        raise ValueError(f"Unknown file type: {fg_filepath}")
    logger.debug(f"Loaded data: {fg_filepath}")
    fg.print_summary()

    solver_params = GtsamSolverParams(
        verbose=True,
        save_results=True,
        init_technique=args.init_technique,
        custom_init_file=args.custom_init_file,
    )
    results_filepath = join(args.results_dir, args.results_filename)
    solve_mle_gtsam(fg, solver_params, results_filepath)
