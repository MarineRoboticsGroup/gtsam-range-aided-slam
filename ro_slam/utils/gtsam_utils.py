import numpy as np
from typing import List, Tuple, Union, Dict
import tqdm  # type: ignore
import re

from gtsam.gtsam import (
    NonlinearFactorGraph,
    Values,
    RangeFactor2D,
    noiseModel,
    BetweenFactorPose2,
    Pose2,
    PosePriorFactorPose2,
)

from py_factor_graph.factor_graph import FactorGraphData
from ro_slam.utils.matrix_utils import (
    _check_square,
    _check_transformation_matrix,
    get_random_vector,
    get_random_rotation_matrix,
    get_random_transformation_matrix,
    get_rotation_matrix_from_theta,
    get_theta_from_rotation_matrix_so_projection,
)
from ro_slam.utils.solver_utils import SolverResults, VariableValues


##### Add costs #####


def add_distances_cost(
    graph: NonlinearFactorGraph,
    data: FactorGraphData,
):
    """Adds in the cost due to the distances as:
    sum_{i,j} k_ij * ||d_ij - d_ij^meas||^2

    Args:
        graph (NonlinearFactorGraph): the graph to add the cost to
        distances (Dict[Tuple[str, str], np.ndarray]): [description]
        data (FactorGraphData): [description]

    """
    for range_measure in data.range_measurements:
        pose_key = range_measure.pose_key
        landmark_key = range_measure.landmark_key

        range_noise = noiseModel.Isotropic.Sigma(1, range_measure.stddev)
        range_factor = RangeFactor2D(
            pose_key, landmark_key, range_measure.dist, range_noise
        )
        graph.push_back(range_factor)


def add_odom_cost(
    graph: NonlinearFactorGraph,
    data: FactorGraphData,
):
    """Add the cost associated with the odometry measurements as:

        translation component of cost
        k_ij * ||t_i - t_j - R_i @ t_ij^meas||^2

        rotation component of cost
        tau_ij * || R_j - (R_i @ R_ij^\top) ||_\frob^2

    Args:
        graph (NonlinearFactorGraph): the graph to add the cost to
        translations (Dict[str, np.ndarray]): the variables representing translations
        rotations (Dict[str, np.ndarray]): the variables representing rotations
        data (FactorGraphData): the factor graph data

    """
    for odom_chain in data.odom_measurements:
        for odom_measure in odom_chain:

            # the indices of the related poses in the odometry measurement
            i_key = odom_measure.base_pose
            j_key = odom_measure.to_pose

            # add the factor to the factor graph
            odom_noise = noiseModel.Diagonal.Sigmas(np.diag(odom_measure.covariance))
            rel_pose = Pose2(odom_measure.x, odom_measure.y, odom_measure.theta)
            odom_factor = BetweenFactorPose2(i_key, j_key, rel_pose, odom_noise)
            graph.push_back(odom_factor)


def add_loop_closure_cost(
    graph: NonlinearFactorGraph,
    data: FactorGraphData,
):
    """Add the cost associated with the loop closure measurements as:

        translation component of cost
        k_ij * ||t_i - t_j - R_i @ t_ij^meas||^2

        rotation component of cost
        tau_ij * || R_j - (R_i @ R_ij^\top) ||_\frob^2

    Args:
        graph (NonlinearFactorGraph): the graph to add the cost to
        translations (Dict[str, np.ndarray]): the variables representing translations
        rotations (Dict[str, np.ndarray]): the variables representing rotations
        data (FactorGraphData): the factor graph data

    """
    for loop_measure in data.loop_closure_measurements:

        # the indices of the related poses in the odometry measurement
        i_key = loop_measure.base_pose
        j_key = loop_measure.to_pose

        loop_noise = noiseModel.Diagonal.Sigmas(np.diag(loop_measure.covariance))
        rel_pose = Pose2(loop_measure.x, loop_measure.y, loop_measure.theta)
        loop_factor = BetweenFactorPose2(i_key, j_key, rel_pose, loop_noise)
        graph.push_back(loop_factor)


##### Initialization strategies #####


def init_pose_variable(init_vals: Values, pose_key: str, pose: np.ndarray):
    """
    Initialize the rotation variables to the given rotation matrix.

    Args:
        rot (np.ndarray): The rotation variables.
        mat (np.ndarray): The rotation matrix.
    """
    _check_square(pose)
    assert pose.shape[0] == 3, "The pose must be a 3x3 matrix"
    # init_vals.insert


def init_landmark_variable(init_vals: Values, lmk_key: str, val: np.ndarray):
    """Initialize the translation variables to the given vector

    Args:
    """
    raise NotImplementedError("Method not implemented")


def set_pose_init_compose(init_vals: Values, data: FactorGraphData) -> None:
    """initializes the rotations by composing the rotations along the odometry chain

    Args:
        rotations (List[np.ndarray]): the rotation variables to initialize
        data (FactorGraphData): the data to use to initialize the rotations
    """
    print("Setting pose initial points by pose composition")

    # iterate over measurements and init the rotations
    for odom_chain in data.odom_measurements:

        # initialize the first rotation to the identity matrix
        curr_pose = np.eye(data.dimension)
        first_pose_name = odom_chain[0].base_pose
        init_pose_variable(init_vals, first_pose_name, curr_pose)

        for odom_measure in odom_chain:

            # update the rotation and initialize the next rotation
            curr_pose = odom_measure.rotation @ curr_pose
            curr_pose_name = odom_measure.to_pose
            init_pose_variable(init_vals, curr_pose_name, curr_pose)


def set_pose_init_gt(
    init_vals: Values,
    data: FactorGraphData,
) -> None:
    """Initialize the translation and rotation variables to the ground truth translation
    variables.

    Args:
        graph (NonlinearFactorGraph): the graph to initialize the variables in
        rotations (Dict[str, np.ndarray]): the rotation variables to initialize
        data (FactorGraphData): the data to use to initialize the variables
    """
    print("Setting pose initial points to ground truth")
    for pose_chain in data.pose_variables:
        for pose_var in pose_chain:
            pose_key = pose_var.name
            true_pose = pose_var.transformation_matrix
            init_pose_variable(init_vals, pose_key, true_pose)


def set_pose_init_random(init_vals: Values, data: FactorGraphData) -> None:
    """Initializes the rotation variables to random.

    Args:
        graph (NonlinearFactorGraph): the graph to initialize the variables in
        rotations (Dict[str, np.ndarray]): the rotation variables to initialize
    """
    print("Setting pose initial points to random")

    for pose_chain in data.pose_variables:
        for pose_var in pose_chain:
            pose_key = pose_var.name
            rand_pose = get_random_transformation_matrix()
            init_pose_variable(init_vals, pose_key, rand_pose)


def set_pose_init_custom(
    init_vals: Values, custom_poses: Dict[str, np.ndarray]
) -> None:
    """[summary]

    Args:
        graph (NonlinearFactorGraph): [description]
        rotations (Dict[str, np.ndarray]): [description]
        custom_rotations (Dict[str, np.ndarray]): [description]
    """
    print("Setting pose initial points to custom")
    for pose_key, pose in custom_poses.items():
        _check_transformation_matrix(pose)
        init_pose_variable(init_vals, pose_key, pose)


def set_landmark_init_gt(
    init_vals: Values,
    data: FactorGraphData,
):
    """Initialize the landmark variables to the ground truth landmarks.

    Args:
        landmarks (Dict[str, np.ndarray]): the landmark variables to initialize
        data (FactorGraphData): the factor graph data to use to initialize the landmarks
    """
    print("Setting landmark initial points to ground truth")
    for true_landmark in data.landmark_variables:

        # get landmark position
        landmark_key = true_landmark.name
        true_pos = np.asarray(true_landmark.true_position)

        # initialize landmark to correct position
        init_landmark_variable(init_vals, landmark_key, true_pos)


def set_landmark_init_random(init_vals: Values, data: FactorGraphData):
    """Initialize the landmark variables to the ground truth landmarks.

    Args:
        graph (NonlinearFactorGraph): the graph to initialize the variables in
        landmarks (Dict[str, np.ndarray]): the landmark variables to initialize
    """
    print("Setting landmark initial points to ground truth")
    for landmark_var in data.landmark_variables:
        landmark_key = landmark_var.name
        rand_vec = get_random_vector(len(landmark_var.true_position))
        init_landmark_variable(init_vals, landmark_key, rand_vec)


def set_landmark_init_custom(
    init_vals: Values,
    custom_landmarks: Dict[str, np.ndarray],
) -> None:
    """[summary]

    Args:
        graph (NonlinearFactorGraph): [description]
        landmarks (Dict[str, np.ndarray]): [description]
        custom_landmarks (Dict[str, np.ndarray]): [description]
    """
    print("Setting landmark initial points to custom")
    for landmark_key, landmark_var in custom_landmarks.items():
        init_landmark_variable(init_vals, landmark_key, landmark_var)


##### Constraints #####


def pin_first_pose(graph: NonlinearFactorGraph, data: FactorGraphData) -> None:
    """
    Pin the first pose of the robot to its true pose.

    Args:
        graph (NonlinearFactorGraph): The graph to pin the pose in
        data (FactorGraphData): The data to use to pin the pose

    """

    # get the first pose variable
    pose = data.pose_variables[0][0]
    pose_key = pose.name
    true_pose = pose.transformation_matrix

    # build the prior noise model
    x_stddev = 0.1
    y_stddev = 0.1
    theta_stddev = 0.1
    prior_uncertainty = noiseModel.Diagonal.Sigmas(
        np.array([x_stddev, y_stddev, theta_stddev])
    )

    # add the prior factor
    pose_prior = PosePriorFactorPose2(pose_key, true_pose, prior_uncertainty)
    graph.push_back(pose_prior)


##### Misc


def get_solved_values(
    result: Values, time: float, data: FactorGraphData
) -> SolverResults:
    """
    Returns the solved values from the result

    Args:
        result (Drake Result Object): the result of the solution
        translations (Dict[str, np.ndarray]): the translation variables
        rotations (Dict[str, np.ndarray]): the rotation variables
        landmarks (Dict[str, np.ndarray]): the landmark variables

    Returns:
        Dict[str, np.ndarray]: the solved translations
        Dict[str, np.ndarray]: the solved rotations
        Dict[str, np.ndarray]: the solved landmarks
        Dict[str, np.ndarray]: the solved distances
    """
    solved_poses: Dict[str, np.ndarray] = {}
    solved_landmarks: Dict[str, np.ndarray] = {}
    solved_distances = None

    for pose_chain in data.pose_variables:
        for pose_var in pose_chain:
            pose_key = pose_var.name
            solved_poses[pose_key] = result.at(pose_key)

    for landmark in data.landmark_variables:
        landmark_key = landmark.name
        solved_landmarks[landmark_key] = result.at(landmark_key)

    return SolverResults(
        VariableValues(
            poses=solved_poses,
            landmarks=solved_landmarks,
            distances=solved_distances,
        ),
        total_time=time,
        solved=True,
    )


def get_symbol_from_name(name: str):
    return name.split("_")[1]
