from typing import List
from os.path import isfile, join, expanduser
import numpy as np
import pickle

from ro_slam.factor_graph.factor_graph import (
    OdomMeasurement,
    RangeMeasurement,
    PoseVariable,
    LandmarkVariable,
    PosePrior,
    LandmarkPrior,
    FactorGraphData,
)


def get_covariance_matrix_from_list(covar_list: List) -> np.ndarray:
    """
    Converts a list of floats to a covariance matrix.

    args:
        covar_list (List): a list of floats representing the covariance matrix

    returns:
        np.ndarray: the covariance matrix
    """
    assert len(covar_list) == 3 * 3, f"{len(covar_list)} != 3x3"
    assert all(isinstance(val, float) for val in covar_list)
    covar_matrix = np.array(
        [
            [covar_list[0], covar_list[1], covar_list[2]],
            [covar_list[3], covar_list[4], covar_list[5]],
            [covar_list[6], covar_list[7], covar_list[8]],
        ]
    )

    assert np.allclose(
        covar_matrix, covar_matrix.T
    ), "Covariance matrix must be symmetric"
    assert covar_matrix.shape == (3, 3), "Covariance matrix must be 3x3"

    return covar_matrix


def parse_efg_factor_graph_file(filepath: str) -> FactorGraphData:
    """
    Parse a factor graph file to extract the factors and variables.

    Args:
        filepath: The path to the factor graph file.

    Returns:
        FactorGraphData: The factor graph data.
    """
    assert isfile(filepath), f"{filepath} is not a file"
    assert filepath.endswith(".fg"), f"{filepath} is not an efg file"

    pose_var_header = "Variable Pose SE2"
    landmark_var_header = "Variable Landmark R2"
    odom_measure_header = "Factor SE2RelativeGaussianLikelihoodFactor"
    range_measure_header = "Factor SE2R2RangeGaussianLikelihoodFactor"
    pose_prior_header = "Factor UnarySE2ApproximateGaussianPriorFactor"
    landmark_prior_header = "Landmark"  # don't have any of these yet

    pose_vars = []
    landmark_vars = []
    odom_measures = []
    range_measures = []
    pose_priors = []
    landmark_priors: List[LandmarkPrior] = []
    with open(filepath, "r") as f:
        for line in f:
            if line.startswith(pose_var_header):
                line_items = line.split()
                pose_name = line_items[3]
                x = float(line_items[4])
                y = float(line_items[5])
                theta = float(line_items[6])
                pose_vars.append(PoseVariable(pose_name, (x, y), theta))
            elif line.startswith(landmark_var_header):
                line_items = line.split()
                landmark_name = line_items[3]
                x = float(line_items[4])
                y = float(line_items[5])
                landmark_vars.append(LandmarkVariable(landmark_name, (x, y)))
            elif line.startswith(odom_measure_header):
                line_items = line.split()
                base_pose = line_items[2]
                local_pose = line_items[3]
                delta_x = float(line_items[4])
                delta_y = float(line_items[5])
                delta_theta = float(line_items[6])
                covar_list = [float(x) for x in line_items[8:]]
                covar = get_covariance_matrix_from_list(covar_list)
                # assert covar[0, 0] == covar[1, 1]
                trans_weight = 1 / (covar[0, 0])
                rot_weight = 1 / (covar[2, 2])
                odom_measures.append(
                    OdomMeasurement(
                        base_pose,
                        local_pose,
                        delta_x,
                        delta_y,
                        delta_theta,
                        trans_weight,
                        rot_weight,
                    )
                )
            elif line.startswith(range_measure_header):
                line_items = line.split()
                var1 = line_items[2]
                var2 = line_items[3]
                dist = float(line_items[4])
                stddev = float(line_items[5])
                range_measures.append(RangeMeasurement(var1, var2, dist, stddev))
            elif line.startswith(pose_prior_header):
                line_items = line.split()
                pose_name = line_items[2]
                x = float(line_items[3])
                y = float(line_items[4])
                theta = float(line_items[5])
                covar_list = [float(x) for x in line_items[7:]]
                covar = get_covariance_matrix_from_list(covar_list)
                pose_priors.append(PosePrior(pose_name, (x, y), theta, covar))
            elif line.startswith(landmark_prior_header):
                raise NotImplementedError

    return FactorGraphData(
        pose_vars,
        landmark_vars,
        odom_measures,
        range_measures,
        pose_priors,
        landmark_priors,
        2,
    )


def parse_pickle_factor_graph_file(filepath: str) -> FactorGraphData:
    """
    Parse a factor graph file to extract the factors and variables.

    Args:
        filepath: The path to the factor graph file.

    Returns:
        FactorGraphData: The factor graph data.
    """
    assert isfile(filepath), f"{filepath} is not a file"
    assert filepath.endswith(".pkl"), f"{filepath} is not a pickle file"
    fg = pickle.load(open(filepath, "rb"))
    return fg


if __name__ == "__main__":
    filepath = expanduser(
        join("~", "data", "example_factor_graphs", "case0", "factor_graph.fg")
    )
    fg = parse_efg_factor_graph_file(filepath)
