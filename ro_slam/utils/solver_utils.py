from typing import Optional, Callable, Union, List
from os.path import isfile
import attr
import logging

logger = logging.getLogger(__name__)

from gtsam.gtsam import (
    ISAM2Params,
    ISAM2DoglegParams,
    ISAM2,
    LevenbergMarquardtOptimizer,
    LevenbergMarquardtParams,
    NonlinearFactorGraph,
    Values,
)

ISAM2_SOLVER = "isam2"
LM_SOLVER = "levenberg_marquardt"
ACCEPTABLE_SOLVERS = [ISAM2_SOLVER, LM_SOLVER]


def solve(
    graph: NonlinearFactorGraph,
    initial_vals: Values,
    solver: str,
) -> Values:
    solver_func = _get_solver_func(solver)
    return solver_func(graph, initial_vals)


def solve_and_return_all_iterates(
    graph: NonlinearFactorGraph,
    initial_vals: Values,
    solver: str,
) -> Values:
    solver_func = _get_solver_func(solver)
    return solver_func(graph, initial_vals, return_all_iterates=True)


def _get_solver_func(solver: str) -> Callable:
    assert solver in ACCEPTABLE_SOLVERS, f"Solver {solver} not in {ACCEPTABLE_SOLVERS}"
    if solver == ISAM2_SOLVER:
        return solve_with_isam2
    elif solver == LM_SOLVER:
        return solve_with_levenberg_marquardt
    else:
        raise ValueError(f"Unknown solver {solver}")


def solve_with_isam2(
    graph: NonlinearFactorGraph, initial_vals: Values, return_all_iterates: bool = False
) -> Union[Values, List[Values]]:
    if return_all_iterates:
        raise NotImplementedError("ISAM2 does not support returning all iterates")

    parameters = ISAM2Params()
    parameters.setOptimizationParams(ISAM2DoglegParams())
    logger.debug(f"ISAM Params: {parameters}")
    isam_solver = ISAM2(parameters)
    isam_solver.update(graph, initial_vals)
    result = isam_solver.calculateEstimate()
    return result


def solve_with_levenberg_marquardt(
    graph: NonlinearFactorGraph, initial_vals: Values, return_all_iterates: bool = False
) -> Union[Values, List[Values]]:
    optimizer = LevenbergMarquardtOptimizer(graph, initial_vals)
    params = LevenbergMarquardtParams()
    params.setVerbosityLM("SUMMARY")

    if not return_all_iterates:
        optimizer.optimize()
        result = optimizer.values()
        return result
    else:
        results = [optimizer.values()]
        while not optimizer.checkConvergence():
            optimizer.iterate()
            results.append(optimizer.values())
        return results
