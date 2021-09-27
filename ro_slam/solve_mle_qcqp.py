import numpy as np
from os.path import expanduser, join
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt  # type: ignore
import scipy.linalg as la  # type: ignore
from gekko import GEKKO as gk  # type: ignore

from ro_slam.qcqp_utils import (
    VarMatrix,
    add_rotation_var,
    add_translation_var,
    pin_first_pose,
    add_pose_variables,
    add_landmark_variables,
    add_distance_variables,
    get_distances_cost,
    get_odom_cost,
    set_rotation_init_gt,
    set_rotation_init_compose,
    set_translation_init_gt,
    set_translation_init_compose,
    set_distance_init_gt,
    set_landmark_init_gt,
)
from ro_slam.factor_graph.parse_factor_graph import parse_factor_graph_file
from ro_slam.factor_graph.factor_graph import FactorGraphData


def solve_mle_problem(data: FactorGraphData):
    """
    Takes the data describing the problem and returns the MLE solution to the
    poses and landmark positions

    args:
        data (FactorGraphData): the data describing the problem
    """
    model = gk(remote=False, name="qcqp")
    obj = 0

    # form objective function
    # Q, D = _get_data_matrix(data)
    # _check_psd(Q)
    # _check_psd(D)
    # full_data_matrix = la.block_diag(Q, D)

    # true_vals = data.true_values_vector

    # _check_psd(full_data_matrix)

    translations, rotations = add_pose_variables(model, data)
    landmarks = add_landmark_variables(model, data)
    distances = add_distance_variables(model, data, translations, landmarks)

    obj += get_distances_cost(distances, data)
    obj += get_odom_cost(translations, rotations, data)

    # add in term to move landmarks to their true locations
    # for i, j in distances.keys():
    #     x_i = translations[i]
    #     l_j = landmarks[j]
    #     diff = x_i - l_j
    # dist_ij = distances[(i, j)]
    # # obj += dist_ij * dist_ij - diff.frob_norm_squared

    # pin first pose at origin
    pin_first_pose(translations[0], rotations[0])

    ### Rotation Initialization
    # set_rotation_init_gt(rotations, data)
    set_rotation_init_compose(rotations, data)

    ### Translation Initialization
    # set_translation_init_gt(translations, data)
    # set_translation_init_compose(translations, data)

    ### Distance Initialization
    set_distance_init_gt(distances, data)

    ### Landmark Initialization
    # set_landmark_init_gt(landmarks, data, model)
    print()

    # perform optimization
    model.setObjective(obj, gk.GRB.MINIMIZE)
    model.optimize()

    # extract the solution
    print()
    print("Landmarks")
    for l in landmarks:
        print(l)
    print()

    print("Translations")
    for t in translations[::10]:
        print(t)
    print()

    # print("Rotations")
    # for r in rotations:
    #     print(r)
    # print()


if __name__ == "__main__":
    file_name = "simEnvironment_grid20x20_rowCorner2_colCorner2_cellScale10_rangeProb05_rangeRadius40_falseRangeProb00_outlierProb01_loopClosureProb01_loopClosureRadius3_falseLoopClosureProb01_timestep1000.fg"

    filepath = expanduser(join("~", "data", "example_factor_graphs", file_name))
    fg = parse_factor_graph_file(filepath)
    solve_mle_problem(fg)
