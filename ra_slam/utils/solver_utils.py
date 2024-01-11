from typing import Callable, Union, List
import logging
import numpy as np

logger = logging.getLogger(__name__)

from gtsam.gtsam import (
    ISAM2Params,
    ISAM2DoglegParams,
    ISAM2,
    LevenbergMarquardtOptimizer,
    LevenbergMarquardtParams,
    NonlinearFactorGraph,
    Values,
    Symbol,
)

ISAM2_SOLVER = "isam2"
LM_SOLVER = "levenberg_marquardt"
ACCEPTABLE_SOLVERS = [ISAM2_SOLVER, LM_SOLVER]


def solve(
    graph: NonlinearFactorGraph,
    initial_vals: Values,
    solver: str,
    return_all_iterates: bool = False,
) -> Union[Values, List[Values]]:
    solver_func = _get_solver_func(solver)
    return solver_func(graph, initial_vals, return_all_iterates=return_all_iterates)


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

    # check that the variables in the graph are all in the initial values
    # otherwise, the optimizer will throw an error
    init_vals_vars = initial_vals.keys()
    graph_vars = graph.keyVector()

    graph_var_set = set(graph_vars)
    init_vals_var_set = set(init_vals_vars)

    in_graph_not_init_vals = graph_var_set - init_vals_var_set
    in_init_vals_not_graph = init_vals_var_set - graph_var_set

    if len(in_graph_not_init_vals) > 0:
        graph_vars_as_symbols = [Symbol(key) for key in in_graph_not_init_vals]
        raise ValueError(
            f"Variables in graph but not in initial values: {graph_vars_as_symbols}"
        )

    if len(in_init_vals_not_graph) > 0:
        init_vals_vars_as_symbols = [Symbol(key) for key in in_init_vals_not_graph]
        raise ValueError(
            f"Variables in initial values but not in graph: {init_vals_vars_as_symbols}"
        )

    if not return_all_iterates:
        optimizer.optimize()
        result = optimizer.values()
        return result
    else:
        results = [optimizer.values()]
        currentError = np.inf
        newError = optimizer.error()
        rel_err_tol = params.getRelativeErrorTol()
        abs_err_tol = params.getAbsoluteErrorTol()
        err_tol = params.getErrorTol()
        max_iter = params.getMaxIterations()

        converged = False
        curr_iter = 0

        while not converged and curr_iter < max_iter:
            optimizer.iterate()
            results.append(optimizer.values())

            currentError = newError
            newError = optimizer.error()

            within_rel_err_tol = (
                abs(newError - currentError) < rel_err_tol * currentError
            )
            within_abs_err_tol = abs(newError - currentError) < abs_err_tol
            within_err_tol = newError < err_tol

            converged = within_rel_err_tol or within_abs_err_tol or within_err_tol
            curr_iter += 1

        return results
