import os
from os.path import join, expanduser
import sys

sys.path.insert(0, os.path.abspath(".."))

from ro_slam.factor_graph.parse_factor_graph import parse_factor_graph_file
from ro_slam.solve_mle_qcqp import solve_mle_problem


if __name__ == "__main__":
    # filepath = expanduser(join("~", "data", "example_factor_graphs", file_name))
    filepath = (
        "/home/alan/data/example_factor_graphs/15_loop_clos/test_9/factor_graph.fg"
    )
    solver = "snopt"
    verbose = False

    fg = parse_factor_graph_file(filepath)
    solve_mle_problem(fg, solver, verbose)
