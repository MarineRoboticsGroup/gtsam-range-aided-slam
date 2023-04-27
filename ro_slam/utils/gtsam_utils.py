import numpy as np
import py
import io
import sys
from contextlib import redirect_stdout, redirect_stderr
from typing import List, Tuple, Union, Dict, Optional
import tqdm  # type: ignore
import re
from py_factor_graph.measurements import (
    PoseMeasurement2D,
    PoseMeasurement3D,
    POSE_MEASUREMENT_TYPES,
)
from py_factor_graph.variables import (
    PoseVariable2D,
    PoseVariable3D,
    POSE_VARIABLE_TYPES,
    LandmarkVariable2D,
    LandmarkVariable3D,
    LANDMARK_VARIABLE_TYPES,
)

VALID_BETWEEN_FACTOR_MODELS = ["SESync", "between"]

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
    Values,
    RangeFactor2D,
    RangeFactor3D,
    RangeFactorPose2,
    RangeFactorPose3,
    noiseModel,
    BetweenFactorPose2,
    BetweenFactorPose3,
    Pose2,
    Pose3,
    Rot3,
    PriorFactorPose2,
    PriorFactorPose3,
    PriorFactorPoint2,
    PriorFactorPoint3,
    symbol,
)
from ro_slam.custom_factors.SESyncFactor2d import RelativePose2dFactor
from ro_slam.custom_factors.SESyncFactor3d import RelativePose3dFactor

from py_factor_graph.factor_graph import FactorGraphData
from py_factor_graph.utils.matrix_utils import (
    _check_transformation_matrix,
    get_random_vector,
    get_random_transformation_matrix,
    get_theta_from_transformation_matrix,
    get_translation_from_transformation_matrix,
    apply_transformation_matrix_perturbation,
)
from py_factor_graph.utils.solver_utils import SolverResults, VariableValues


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
        pose_symbol = get_symbol_from_name(range_measure.pose_key)
        landmark_symbol = get_symbol_from_name(range_measure.landmark_key)

        range_noise = noiseModel.Isotropic.Sigma(1, range_measure.variance)

        # If the landmark is actually secretly a pose, then we use RangeFactorPose2
        if "L" not in range_measure.landmark_key:
            if data.dimension == 2:
                range_factor = RangeFactorPose2(
                    pose_symbol, landmark_symbol, range_measure.dist, range_noise
                )
            elif data.dimension == 3:
                range_factor = RangeFactorPose3(
                    pose_symbol, landmark_symbol, range_measure.dist, range_noise
                )
            else:
                raise ValueError(f"Unknown dimension: {data.dimension}")
        else:
            if data.dimension == 2:
                range_factor = RangeFactor2D(
                    pose_symbol, landmark_symbol, range_measure.dist, range_noise
                )
            elif data.dimension == 3:
                range_factor = RangeFactor3D(
                    pose_symbol, landmark_symbol, range_measure.dist, range_noise
                )
            else:
                raise ValueError(f"Unknown dimension: {data.dimension}")
        graph.push_back(range_factor)


def add_odom_cost(
    graph: NonlinearFactorGraph,
    data: FactorGraphData,
    factor_type: str = "between",
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
    assert (
        factor_type in VALID_BETWEEN_FACTOR_MODELS
    ), f"Unknown factor type: {factor_type}. Valid types are: {VALID_BETWEEN_FACTOR_MODELS}"
    for odom_chain in data.odom_measurements:
        for odom_measure in odom_chain:

            # the indices of the related poses in the odometry measurement
            i_symbol = get_symbol_from_name(odom_measure.base_pose)
            j_symbol = get_symbol_from_name(odom_measure.to_pose)

            # add the factor to the factor graph
            odom_factor = get_pose_to_pose_factor(
                odom_measure, i_symbol, j_symbol, factor_type
            )
            graph.push_back(odom_factor)


def _get_between_factor(
    odom_measure: POSE_MEASUREMENT_TYPES, i_sym: int, j_sym: int
) -> Union[BetweenFactorPose2, BetweenFactorPose3]:
    odom_noise = noiseModel.Diagonal.Sigmas(np.diag(odom_measure.covariance))
    rel_pose = get_relative_pose_from_odom_measure(odom_measure)
    if isinstance(odom_measure, PoseMeasurement2D):
        odom_factor = BetweenFactorPose2(i_sym, j_sym, rel_pose, odom_noise)
    elif isinstance(odom_measure, PoseMeasurement3D):
        odom_factor = BetweenFactorPose3(i_sym, j_sym, rel_pose, odom_noise)
    else:
        raise ValueError(f"Unknown measurement type: {type(odom_measure)}")
    return odom_factor


def _get_between_se_sync_factor(
    odom_measure: POSE_MEASUREMENT_TYPES, i_sym: int, j_sym: int
) -> Union[RelativePose2dFactor, RelativePose3dFactor]:
    if isinstance(odom_measure, PoseMeasurement2D):
        odom_factor = RelativePose2dFactor(
            i_sym,
            j_sym,
            odom_measure.rotation_matrix,
            odom_measure.translation_vector,
            odom_measure.rotation_precision,
            odom_measure.translation_precision,
        )
    elif isinstance(odom_measure, PoseMeasurement3D):
        odom_factor = RelativePose3dFactor(
            i_sym,
            j_sym,
            odom_measure.rotation_matrix,
            odom_measure.translation_vector,
            odom_measure.rotation_precision,
            odom_measure.translation_precision,
        )
    else:
        raise ValueError(f"Unknown measurement type: {type(odom_measure)}")
    return odom_factor


def get_pose_to_pose_factor(
    odom_measure: POSE_MEASUREMENT_TYPES,
    i_symbol: int,
    j_symbol: int,
    factor_model: str = "SESync",
) -> Union[BetweenFactorPose2, BetweenFactorPose3]:
    """Get the odometry factor from the odometry measurement.

    Args:
        odom_measure (POSE_MEASUREMENT_TYPES): the odometry measurement
        i_symbol (int): the symbol for the first pose
        j_symbol (int): the symbol for the second pose

    Returns:
        Union[BetweenFactorPose2, BetweenFactorPose3]: the relative pose factor
    """
    assert (
        factor_model in VALID_BETWEEN_FACTOR_MODELS
    ), f"Invalid factor model: {factor_model}. Valid models are: {VALID_BETWEEN_FACTOR_MODELS}"
    if factor_model == "SESync":
        odom_factor = _get_between_se_sync_factor(odom_measure, i_symbol, j_symbol)
    elif factor_model == "between":
        odom_factor = _get_between_factor(odom_measure, i_symbol, j_symbol)
    else:
        raise ValueError(f"Unknown factor model: {factor_model}")
    return odom_factor


#    return odom_factor


def get_relative_pose_from_odom_measure(odom_measure: POSE_MEASUREMENT_TYPES):
    """Get the relative pose from the odometry measurement.

    Args:
        odom_measure (POSE_MEASUREMENT_TYPES): the odometry measurement

    Returns:
        Pose2: the relative pose
    """
    if isinstance(odom_measure, PoseMeasurement2D):
        return Pose2(odom_measure.x, odom_measure.y, odom_measure.theta)
    elif isinstance(odom_measure, PoseMeasurement3D):
        return Pose3(
            Rot3(odom_measure.rotation_matrix), odom_measure.translation_vector
        )
    else:
        err = f"Unknown odometry measurement type: {type(odom_measure)}"
        logger.error(err)
        raise ValueError(err)


def add_loop_closure_cost(
    graph: NonlinearFactorGraph,
    data: FactorGraphData,
    factor_type: str = "between",
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
    assert (
        factor_type in VALID_BETWEEN_FACTOR_MODELS
    ), f"Invalid factor model: {factor_type}. Valid models are: {VALID_BETWEEN_FACTOR_MODELS}"
    for loop_measure in data.loop_closure_measurements:

        # the indices of the related poses in the odometry measurement
        i_symbol = get_symbol_from_name(loop_measure.base_pose)
        j_symbol = get_symbol_from_name(loop_measure.to_pose)
        loop_factor = get_pose_to_pose_factor(
            loop_measure, i_symbol, j_symbol, factor_type
        )
        graph.push_back(loop_factor)


def add_landmark_prior_cost(
    graph: NonlinearFactorGraph,
    data: FactorGraphData,
):
    """Add the cost associated with the landmark priors.

    Args:
        graph (NonlinearFactorGraph): the graph to add the cost to
        data (FactorGraphData): the factor graph data

    """
    for landmark_prior in data.landmark_priors:
        landmark_symbol = get_symbol_from_name(landmark_prior.name)
        landmark_noise = noiseModel.Diagonal.Sigmas(np.diag(landmark_prior.covariance))
        if data.dimension == 2:
            landmark_prior_factor = PriorFactorPoint2(
                landmark_symbol, landmark_prior.position, landmark_noise
            )
        elif data.dimension == 3:
            raise NotImplementedError("3D landmark priors not implemented")
            landmark_prior_factor = PriorFactorPoint3(
                landmark_symbol, landmark_prior.position, landmark_noise
            )
        else:
            raise ValueError(f"Unknown dimension: {data.dimension}")
        graph.push_back(landmark_prior_factor)


##### Initialization strategies #####


def init_pose_variable(init_vals: Values, pose_key: str, val: np.ndarray, dim: int):
    """
    Initialize the rotation variables to the given rotation matrix.

    Args:
        rot (np.ndarray): The rotation variables.
        mat (np.ndarray): The rotation matrix.
    """
    assert dim in [2, 3], f"Invalid dimension: {dim}"
    _check_transformation_matrix(val, dim=dim)
    pose_symbol = get_symbol_from_name(pose_key)
    if dim == 2:
        pose = get_pose2_from_matrix(val)
    elif dim == 3:
        pose = get_pose3_from_matrix(val)

    init_vals.insert(pose_symbol, pose)


def init_landmark_variable(init_vals: Values, lmk_key: str, val: np.ndarray):
    """Initialize the translation variables to the given vector

    Args:
        init_vals (Values): the initial values
        lmk_key (str): the key for the landmark
        val (np.ndarray): the value to initialize the landmark to
    """
    dim = val.shape[0]
    assert dim in [2, 3], f"Invalid landmark dimension: {dim}"
    landmark_symbol = get_symbol_from_name(lmk_key)
    init_vals.insert(landmark_symbol, val)


def set_pose_init_compose(
    init_vals: Values,
    data: FactorGraphData,
    gt_start: bool = False,
    perturb_magnitude: Optional[float] = None,
    perturb_rotation: Optional[float] = None,
) -> None:
    """initializes the rotations by composing the rotations along the odometry chain

    Args:
        rotations (List[np.ndarray]): the rotation variables to initialize
        data (FactorGraphData): the data to use to initialize the rotations
        gt_start (bool, optional): whether to use the ground truth start pose.
        Otherwise, will use random start pose
        perturb_magnitude (float, optional): the magnitude of the perturbation
        perturb_rotation (float, optional): the magnitude of the perturbation
    """
    logger.debug("Setting pose initial points by pose composition")

    if not gt_start and data.num_robots > 1:
        logger.warning("Using random start pose - this is not ground truth start")

    # iterate over measurements and init the rotations
    for robot_idx, odom_chain in enumerate(data.odom_measurements):
        if len(odom_chain) == 0:
            continue  # Skip empty pose chains
        # initialize the first rotation to the identity matrix
        if gt_start or robot_idx == 0:
            curr_pose = data.pose_variables[robot_idx][0].transformation_matrix
        else:
            curr_pose = get_random_transformation_matrix(dim=data.dimension)
        first_pose_name = odom_chain[0].base_pose

        # if we have perturbation parameters, then we perturb the first pose
        if perturb_magnitude is not None and perturb_rotation is not None:
            logger.warning("Perturbing the first pose")
            curr_pose = apply_transformation_matrix_perturbation(
                curr_pose, perturb_magnitude, perturb_rotation
            )

        init_pose_variable(init_vals, first_pose_name, curr_pose, dim=data.dimension)

        for odom_measure in odom_chain:

            # update the rotation and initialize the next rotation
            curr_pose = curr_pose @ odom_measure.transformation_matrix

            # if we have perturbation parameters, then we perturb the pose
            # if perturb_magnitude is not None and perturb_rotation is not None:
            #     curr_pose = apply_transformation_matrix_perturbation(
            #         curr_pose, perturb_magnitude, perturb_rotation
            #     )

            curr_pose_name = odom_measure.to_pose
            init_pose_variable(init_vals, curr_pose_name, curr_pose, dim=data.dimension)


def set_pose_init_gt(
    init_vals: Values,
    data: FactorGraphData,
    perturb_magnitude: Optional[float] = None,
    perturb_rotation: Optional[float] = None,
) -> None:
    """Initialize the translation and rotation variables to the ground truth
    translation values.

    Args:
        graph (NonlinearFactorGraph): the graph to initialize the variables in
        rotations (Dict[str, np.ndarray]): the rotation variables to initialize
        data (FactorGraphData): the data to use to initialize the variables
        perturb_magnitude (Optional[float]): the magnitude of the perturbation
        to apply
        perturb_rotation (Optional[float]): the rotation of the perturbation
    """
    logger.debug("Setting pose initial points to ground truth")
    for pose_chain in data.pose_variables:
        for pose_idx, pose_var in enumerate(pose_chain):
            pose_key = pose_var.name
            true_pose = pose_var.transformation_matrix

            # if we have perturbation parameters, then we perturb the first pose
            if perturb_magnitude is not None and perturb_rotation is not None:
                logger.warning("Perturbing the first pose")
                true_pose = apply_transformation_matrix_perturbation(
                    true_pose, perturb_magnitude, perturb_rotation
                )

            init_pose_variable(init_vals, pose_key, true_pose, dim=data.dimension)


def set_pose_init_random(init_vals: Values, data: FactorGraphData) -> None:
    """Initializes the pose variables to random.

    Args:
        graph (NonlinearFactorGraph): the graph to initialize the variables in
        rotations (Dict[str, np.ndarray]): the rotation variables to initialize
    """
    logger.debug("Setting pose initial points to random")

    for pose_chain in data.pose_variables:
        for pose_var in pose_chain:
            pose_key = pose_var.name
            rand_pose = get_random_transformation_matrix(dim=data.dimension)
            init_pose_variable(init_vals, pose_key, rand_pose, dim=data.dimension)


def set_pose_init_custom(
    init_vals: Values, custom_poses: Dict[str, np.ndarray]
) -> None:
    """[summary]

    Args:
        graph (NonlinearFactorGraph): [description]
        rotations (Dict[str, np.ndarray]): [description]
        custom_rotations (Dict[str, np.ndarray]): [description]
    """
    logger.debug("Setting pose initial points to custom")
    for pose_key, pose in custom_poses.items():
        _check_transformation_matrix(pose, dim=pose.shape[0] - 1)
        init_pose_variable(init_vals, pose_key, pose, dim=pose.shape[0] - 1)


def set_landmark_init_gt(
    init_vals: Values,
    data: FactorGraphData,
):
    """Initialize the landmark variables to the ground truth landmarks.

    Args:
        landmarks (Dict[str, np.ndarray]): the landmark variables to initialize
        data (FactorGraphData): the factor graph data to use to initialize the landmarks
    """
    logger.debug("Setting landmark initial points to ground truth")
    for true_landmark in data.landmark_variables:

        # get landmark position
        landmark_key = true_landmark.name
        true_pos = np.asarray(true_landmark.true_position)

        # initialize landmark to correct position
        init_landmark_variable(init_vals, landmark_key, true_pos)


def set_landmark_init_random(init_vals: Values, data: FactorGraphData):
    """Initialize the landmark variables to random values.

    Args:
        graph (NonlinearFactorGraph): the graph to initialize the variables in
        landmarks (Dict[str, np.ndarray]): the landmark variables to initialize
    """
    logger.debug("Setting landmark initial points to random values")
    for landmark_var in data.landmark_variables:
        landmark_key = landmark_var.name

        x_min = data.x_min - 50
        x_max = data.x_max + 50

        y_min = data.y_min - 50
        y_max = data.y_max + 50

        if data.dimension == 2:
            rand_vec = np.array([x_min, y_min])
            # rand_vec = get_random_vector(dim=2, bounds=[x_min, x_max, y_min, y_max])
        elif data.dimension == 3:
            z_min = data.z_min - 50
            z_max = data.z_max + 50
            rand_vec = np.array([x_min, y_min, z_min])
            # rand_vec = get_random_vector(
            #     dim=3, bounds=[x_min, x_max, y_min, y_max, z_min, z_max]
            # )

        # rand_vec = np.zeros(data.dimension) # ZEROS looks good for us with goats_14
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
    logger.debug("Setting landmark initial points to custom")
    for landmark_key, landmark_var in custom_landmarks.items():
        init_landmark_variable(init_vals, landmark_key, landmark_var)


##### Constraints #####


def pin_first_pose(graph: NonlinearFactorGraph, data: FactorGraphData) -> None:
    """
    Pin the first pose of the robot to its true pose.
    Also pins the landmark to the first pose.

    Args:
        graph (NonlinearFactorGraph): The graph to pin the pose in
        data (FactorGraphData): The data to use to pin the pose

    """
    # build the prior noise model
    trans_stddev = 0.1
    rot_stddev = 0.05
    if data.dimension == 2:
        prior_uncertainty = noiseModel.Diagonal.Sigmas(
            np.array([trans_stddev ** 2] * 2 + [rot_stddev ** 2])
        )
    elif data.dimension == 3:
        prior_uncertainty = noiseModel.Diagonal.Sigmas(
            np.array([trans_stddev ** 2] * 3 + [rot_stddev ** 2] * 3)
        )
    else:
        raise ValueError(f"The factor graph dimension is bad! D={data.dimension}")

    for pose_chain in data.pose_variables:
        if len(pose_chain) == 0:
            continue

        # get the first pose variable
        pose = pose_chain[0]

        # add the prior factor
        pose_prior = get_gtsam_prior_from_pose_variable(pose, prior_uncertainty)
        graph.push_back(pose_prior)

        return  # TODO: Pin only the first pose, remove if not needed


def get_gtsam_pose_from_pose_variable(
    pose_var: POSE_VARIABLE_TYPES,
) -> Union[Pose2, Pose3]:
    if isinstance(pose_var, PoseVariable2D):
        return Pose2(pose_var.true_x, pose_var.true_y, pose_var.true_theta)
    elif isinstance(pose_var, PoseVariable3D):
        return Pose3(Rot3(pose_var.true_rotation), pose_var.position_vector)
    else:
        err = f"Invalid pose variable type: {type(pose_var)}"
        logger.error(err)
        raise ValueError(err)


def get_gtsam_prior_from_pose_variable(
    pose_var: POSE_VARIABLE_TYPES, prior_uncertainty: np.ndarray
) -> Union[PriorFactorPose2, PriorFactorPose3]:
    pose_symbol = get_symbol_from_name(pose_var.name)
    true_pose = get_gtsam_pose_from_pose_variable(pose_var)
    if isinstance(pose_var, PoseVariable2D):
        return PriorFactorPose2(pose_symbol, true_pose, prior_uncertainty)
    elif isinstance(pose_var, PoseVariable3D):
        return PriorFactorPose3(pose_symbol, true_pose, prior_uncertainty)
    else:
        err = f"Invalid pose variable type: {type(pose_var)}"
        logger.error(err)
        raise ValueError(err)


def pin_first_landmark(graph: NonlinearFactorGraph, data: FactorGraphData) -> None:
    """
    Pin the first pose of the robot to its true pose.
    Also pins the landmark to the first pose.

    Args:
        graph (NonlinearFactorGraph): The graph to pin the pose in
        data (FactorGraphData): The data to use to pin the pose

    """
    x_stddev = 0.1
    y_stddev = 0.1
    prior_pt2_uncertainty = noiseModel.Diagonal.Sigmas(
        np.array([x_stddev ** 2, y_stddev ** 2])
    )

    for landmark_var in data.landmark_variables:
        landmark_symbol = get_symbol_from_name(landmark_var.name)
        landmark_point = np.array(
            [landmark_var.true_position[0], landmark_var.true_position[1]]
        )
        landmark_prior = PriorFactorPoint2(
            landmark_symbol, landmark_point, prior_pt2_uncertainty
        )
        graph.push_back(landmark_prior)

        break  # TODO: Pin only the first landmark, remove if not needed


##### Misc


def generate_detailed_report_of_factor_costs(
    graph: NonlinearFactorGraph, values: Values
) -> None:
    """[summary]

    Args:
        graph (NonlinearFactorGraph): [description]
        values (Values): [description]
    """
    # call printErrors to and save the output to a string
    capture = py.io.StdCaptureFD()
    graph.printErrors(values)
    output, error = capture.reset()
    split_output = output.splitlines()

    for idx in range(len(split_output)):
        if "RangeFactor" in split_output[idx]:
            info = [split_output[idx + cnt] for cnt in [0, 1, 3]]
            print("\n".join(info), end="\n\n")

    # get the symbol for A95 and A96
    a95_symbol = get_symbol_from_name("A95")
    a96_symbol = get_symbol_from_name("A96")

    # print the values of A95 and A96
    print(f"A95: {values.atPose2(a95_symbol)}")
    print(f"A96: {values.atPose2(a96_symbol)}")

    # print the value of A0
    a0_symbol = get_symbol_from_name("A0")
    print(f"A0: {values.atPose2(a0_symbol)}")

    # print the value of L0
    l0_symbol = get_symbol_from_name("L0")
    print(f"L0: {values.atPoint2(l0_symbol)}")


def get_factor_graph_from_pyfg_data(data: FactorGraphData) -> NonlinearFactorGraph:

    factor_graph = NonlinearFactorGraph()

    # form objective function
    add_distances_cost(factor_graph, data)
    add_odom_cost(factor_graph, data)
    add_loop_closure_cost(factor_graph, data)
    add_landmark_prior_cost(factor_graph, data)

    # pin first pose at origin
    pin_first_pose(factor_graph, data)

    return factor_graph


def get_gtsam_values_from_variable_values(values: VariableValues, dim: int) -> Values:
    """[summary]

    Args:
        values (VariableValues): [description]
        dim (int): [description]

    Returns:
        Values: [description]
    """
    gtsam_values = Values()
    assert dim in [2, 3], f"Invalid dimension: {dim}"

    for pose_key, pose_val in values.poses.items():
        _check_transformation_matrix(pose_val, dim=dim)
        if dim == 2:
            gtsam_pose = get_pose2_from_matrix(pose_val)
        elif dim == 3:
            gtsam_pose = get_pose3_from_matrix(pose_val)
        pose_symbol = get_symbol_from_name(pose_key)
        gtsam_values.insert(pose_symbol, gtsam_pose)

    for landmark_key, landmark_val in values.landmarks.items():
        assert (
            dim == landmark_val.shape[0]
        ), f"Invalid landmark dimension: {landmark_val.shape}"
        landmark_symbol = get_symbol_from_name(landmark_key)
        gtsam_values.insert(landmark_symbol, landmark_val)

    return gtsam_values


def get_cost_at_variable_values(pyfg: FactorGraphData, values: VariableValues) -> float:
    """[summary]

    Args:
        pyfg (FactorGraphData): [description]
        values (VariableValues): [description]

    Returns:
        float: [description]
    """
    graph = get_factor_graph_from_pyfg_data(pyfg)
    gtsam_values = get_gtsam_values_from_variable_values(values, pyfg.dimension)
    cost = graph.error(gtsam_values)
    return cost


def get_solved_values(
    result: Values, time: float, data: FactorGraphData, cost: float
) -> SolverResults:
    """
    Returns the solved values from the result

    Args:
        result (Values): The result from the solver
        time (float): The time it took to solve the graph
        data (FactorGraphData): The data used to formulate the problem
        cost (float): The cost of the solution

    Returns:
        SolverResults: The results of the solver
    """
    solved_poses: Dict[str, np.ndarray] = {}
    solved_landmarks: Dict[str, np.ndarray] = {}
    solved_distances: Dict[Tuple[str, str], float] = {}
    dim = data.dimension

    def _load_pose_result_to_solved_poses(pose_var: POSE_VARIABLE_TYPES) -> None:
        pose_symbol = get_symbol_from_name(pose_var.name)
        if isinstance(pose_var, PoseVariable2D):
            pose_result = result.atPose2(pose_symbol)
        elif isinstance(pose_var, PoseVariable3D):
            pose_result = result.atPose3(pose_symbol)
        pose_mat = pose_result.matrix()
        solved_poses[pose_var.name] = pose_mat

    def _load_landmark_result_to_solved_landmarks(
        landmark_var: LANDMARK_VARIABLE_TYPES,
    ) -> None:
        landmark_symbol = get_symbol_from_name(landmark_var.name)
        if isinstance(landmark_var, LandmarkVariable2D):
            landmark_result = result.atPoint2(landmark_symbol)
        elif isinstance(landmark_var, LandmarkVariable3D):
            landmark_result = result.atPoint3(landmark_symbol)
        solved_landmarks[landmark_var.name] = landmark_result

    def _load_distance_result_to_solved_distances(assoc: Tuple[str, str]):
        trans1 = solved_poses[assoc[0]][:dim, -1]

        if assoc[1] in solved_poses:
            trans2 = solved_poses[assoc[1]][:dim, -1]
        elif assoc[1] in solved_landmarks:
            trans2 = solved_landmarks[assoc[1]]
        else:
            raise ValueError(f"Invalid association: {assoc}")

        diff = trans1 - trans2
        solved_distances[assoc] = diff / np.linalg.norm(diff)

    for pose_chain in data.pose_variables:
        for pose_var in pose_chain:
            _load_pose_result_to_solved_poses(pose_var)

    for landmark in data.landmark_variables:
        _load_landmark_result_to_solved_landmarks(landmark)

    for range_measurement in data.range_measurements:
        association = range_measurement.association
        _load_distance_result_to_solved_distances(association)

    return SolverResults(
        VariableValues(
            dim=data.dimension,
            poses=solved_poses,
            landmarks=solved_landmarks,
            distances=solved_distances,
        ),
        total_time=time,
        solved=True,
        pose_chain_names=data.get_pose_chain_names(),
        solver_cost=cost,
    )


def get_symbol_from_name(name: str) -> symbol:
    """
    Returns the symbol from a variable name
    """
    assert isinstance(name, str)
    capital_letters_pattern = re.compile(r"^[A-Z][0-9]+$")
    assert capital_letters_pattern.match(name) is not None, "Invalid name"

    return symbol(name[0], int(name[1:]))


def get_pose2_from_matrix(pose_matrix: np.ndarray) -> Pose2:
    """
    Returns the pose2 from a transformation matrix
    """
    _check_transformation_matrix(pose_matrix, dim=2)
    theta = get_theta_from_transformation_matrix(pose_matrix)
    trans = get_translation_from_transformation_matrix(pose_matrix)
    return Pose2(trans[0], trans[1], theta)


def get_pose3_from_matrix(pose_matrix: np.ndarray) -> Pose3:
    """_summary_

    Args:
        pose_matrix (np.ndarray): _description_

    Returns:
        Pose3: _description_
    """
    _check_transformation_matrix(pose_matrix, dim=3)
    rot_matrix = pose_matrix[:3, :3]
    tx, ty, tz = pose_matrix[:3, 3]
    return Pose3(Rot3(rot_matrix), np.array([tx, ty, tz]))
