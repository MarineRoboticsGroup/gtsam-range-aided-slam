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

from gtsam.gtsam import NonlinearFactorGraph, Values
from typing import List, Union

from py_factor_graph.factor_graph import FactorGraphData
from py_factor_graph.utils.solver_utils import (
    SolverResults,
)

from ra_slam.utils.gtsam_utils import GtsamSolverParams
from ra_slam.utils.solver_utils import solve, ISAM2_SOLVER
import ra_slam.utils.gtsam_utils as gt_ut


def solve_mle_gtsam(
    data: FactorGraphData,
    solver_params: GtsamSolverParams,
    solver: str = ISAM2_SOLVER,
    return_all_iterates: bool = False,
) -> Union[SolverResults, List[SolverResults]]:
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
    initial_values = gt_ut.get_initial_values(solver_params, data)
    gt_ut.pin_first_pose(factor_graph, data)

    start_time = time.perf_counter()
    gtsam_result = solve(
        factor_graph, initial_values, solver, return_all_iterates=return_all_iterates
    )
    tot_time = time.perf_counter() - start_time

    # return the results
    if return_all_iterates:
        assert isinstance(
            gtsam_result, list
        ), f"Expected list, got {type(gtsam_result)}"
        results = [
            gt_ut.get_solved_values(result, tot_time, data) for result in gtsam_result
        ]
        return results
    else:
        assert isinstance(gtsam_result, Values)
        res = gt_ut.get_solved_values(gtsam_result, tot_time, data)
        return res
