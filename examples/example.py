import os
from os.path import join, expanduser
import sys

sys.path.insert(0, os.path.abspath(".."))

from ro_slam.factor_graph.parse_factor_graph import parse_factor_graph_file
from ro_slam.solve_mle_qcqp import solve_mle_problem


if __name__ == "__main__":
    file_name = "simEnvironment_grid30x30_rowCorner5_colCorner5_cellScale10_rangeProb05_rangeRadius40_falseRangeProb00_outlierProb01_loopClosureProb01_loopClosureRadius3_falseLoopClosureProb01_timestep1000.fg"
    file_name = "simEnvironment_grid30x30_rowCorner5_colCorner5_cellScale10_rangeProb05_rangeRadius40_falseRangeProb00_outlierProb01_loopClosureProb01_loopClosureRadius3_falseLoopClosureProb01_timestep20.fg"
    # file_name = "simEnvironment_grid30x30_rowCorner5_colCorner5_cellScale10_rangeProb05_rangeRadius40_falseRangeProb00_outlierProb01_loopClosureProb01_loopClosureRadius3_falseLoopClosureProb01_timestep100.fg"

    filepath = expanduser(join("~", "data", "example_factor_graphs", file_name))
    fg = parse_factor_graph_file(filepath)
    solve_mle_problem(fg)
