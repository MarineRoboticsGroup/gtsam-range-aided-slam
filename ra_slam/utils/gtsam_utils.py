import numpy as np
from typing import Tuple, Union, Dict, Optional
import re
from py_factor_graph.measurements import (
    PoseMeasurement2D,
    PoseMeasurement3D,
    POSE_MEASUREMENT_TYPES,
    PoseToLandmarkMeasurement2D,
    PoseToLandmarkMeasurement3D,
)
from py_factor_graph.variables import (
    PoseVariable2D,
    PoseVariable3D,
    POSE_VARIABLE_TYPES,
    LandmarkVariable2D,
    LandmarkVariable3D,
    LANDMARK_VARIABLE_TYPES,
)
from py_factor_graph.utils.name_utils import get_time_idx_from_frame_name

VALID_BETWEEN_FACTOR_MODELS = ["SESync", "between"]

import logging, coloredlogs
from attrs import define, field
from os.path import isfile

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
    Rot2,
    Rot3,
    PriorFactorPose2,
    PriorFactorPose3,
    PriorFactorPoint2,
    PriorFactorPoint3,
    symbol,
)

try:
    from gtsam import SESyncFactor2d as RelativePose2dFactor

    logger.debug("Found C++ SESyncFactor2d")
except ImportError:
    logger.warning("Using python SESyncFactor2d - will be much slower")
    from ra_slam.custom_factors.SESyncFactor2d import RelativePose2dFactor

try:
    from gtsam import SESyncFactor3d as RelativePose3dFactor

    logger.debug("Found C++ SESyncFactor3d")
except ImportError:
    logger.warning("Using python SESyncFactor3d - will be much slower")
    from ra_slam.custom_factors.SESyncFactor3d import RelativePose3dFactor

try:
    # gtu.PoseToPointFactor2D, 3D
    from gtsam_unstable import PoseToPointFactor2D as PoseToPoint2dFactor
    from gtsam_unstable import PoseToPointFactor3D as PoseToPoint3dFactor

    logger.info("Found C++ PoseToPointFactor for 2D and 3D")
except ImportError:
    logger.warning("Using python PoseToPointFactor - will be much slower")
    from ra_slam.custom_factors.PoseToPointFactor import (
        PoseToPoint2dFactor,
        PoseToPoint3dFactor,
    )

from py_factor_graph.factor_graph import FactorGraphData
from py_factor_graph.utils.matrix_utils import (
    _check_transformation_matrix,
    get_random_vector,
    get_random_transformation_matrix,
    get_theta_from_transformation_matrix,
    get_theta_from_rotation_matrix,
    get_translation_from_transformation_matrix,
    apply_transformation_matrix_perturbation,
)
from py_factor_graph.utils.solver_utils import (
    SolverResults,
    VariableValues,
    load_custom_init_file,
)


@define(frozen=True)
class GtsamSolverParams:
    init_technique: str = field()
    landmark_init: str = field()
    custom_init_file: Optional[str] = field(default=None)
    init_translation_perturbation: Optional[float] = field(default=None)
    init_rotation_perturbation: Optional[float] = field(default=None)
    start_at_gt: bool = field(default=False)

    @init_technique.validator
    def _check_init_technique(self, attribute, value):
        init_options = ["gt", "compose", "random", "custom"]
        if value not in init_options:
            raise ValueError(
                f"init_technique must be one of {init_options}, not {value}"
            )

    @landmark_init.validator
    def _check_landmark_init(self, attribute, value):
        init_options = ["gt", "random", "custom"]
        if value not in init_options:
            raise ValueError(
                f"landmark_init must be one of {init_options}, not {value}"
            )

    @custom_init_file.validator
    def _check_custom_init_file(self, attribute, value):
        if value is not None:
            if not isfile(value):
                raise ValueError(f"custom_init_file {value} does not exist")


##### Add costs #####


def add_all_costs(
    graph: NonlinearFactorGraph,
    data: FactorGraphData,
    rel_pose_factor_type: str = "SESync",
    # rel_pose_factor_type: str = "between",
) -> None:
    # form objective function
    add_distances_cost(graph, data)
    add_odom_cost(graph, data, factor_type=rel_pose_factor_type)
    add_loop_closure_cost(graph, data, factor_type=rel_pose_factor_type)
    add_pose_landmark_cost(graph, data)
    add_landmark_prior_cost(graph, data)


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
        pose_symbol = get_symbol_from_name(range_measure.first_key)
        landmark_symbol = get_symbol_from_name(range_measure.second_key)

        range_noise = noiseModel.Isotropic.Sigma(1, range_measure.variance / 4)

        # If the landmark is actually secretly a pose, then we use RangeFactorPose2
        if "L" not in range_measure.second_key:
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
    factor_type: str = "SESync",
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
        rot2_measure = Rot2(get_theta_from_rotation_matrix(odom_measure.rotation_matrix))
        odom_factor = RelativePose2dFactor(
            i_sym,
            j_sym,
            rot2_measure,
            odom_measure.translation_vector,
            0.5 * odom_measure.rotation_precision,
            0.5 * odom_measure.translation_precision,
        )
    elif isinstance(odom_measure, PoseMeasurement3D):
        rot3_measure = Rot3(odom_measure.rotation_matrix)
        odom_factor = RelativePose3dFactor(
            i_sym,
            j_sym,
            rot3_measure,
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
    factor_model: str = "between",
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
    factor_type: str = "SESync",
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


def add_pose_landmark_cost(
    graph: NonlinearFactorGraph,
    data: FactorGraphData,
):
    """Add the cost associated with the loop closure measurements as:

        translation component of cost
        k_ij * ||t_i - t_j - R_i @ t_ij^meas||^2

    Args:
        graph (NonlinearFactorGraph): the graph to add the cost to
        data (FactorGraphData): the factor graph data
    """
    for plm in data.pose_landmark_measurements:
        # the indices of the related poses in the odometry measurement
        i_symbol = get_symbol_from_name(plm.pose_name)
        j_symbol = get_symbol_from_name(plm.landmark_name)
        if isinstance(plm, PoseToLandmarkMeasurement2D):
            noise_model = noiseModel.Diagonal.Precisions(
                2 * np.array([plm.translation_precision] * 2)
            )
            plm_factor = PoseToPoint2dFactor(
                i_symbol, j_symbol, plm.translation_vector, noise_model
            )
        elif isinstance(plm, PoseToLandmarkMeasurement3D):
            noise_model = noiseModel.Diagonal.Precisions(
                2 * np.array([plm.translation_precision] * 3)
            )
            plm_factor = PoseToPoint3dFactor(
                i_symbol, j_symbol, plm.translation_vector, noise_model
            )
        else:
            raise ValueError(f"Unknown measurement type: {type(plm)}")
        graph.push_back(plm_factor)


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


def get_initial_values(
    solver_params: GtsamSolverParams, data: FactorGraphData
) -> Values:
    initial_values = Values()
    if solver_params.init_technique == "gt":
        set_pose_init_gt(
            initial_values,
            data,
            solver_params.init_translation_perturbation,
            solver_params.init_rotation_perturbation,
        )
    elif solver_params.init_technique == "compose":
        set_pose_init_compose(
            initial_values,
            data,
            gt_start=solver_params.start_at_gt,
            perturb_magnitude=solver_params.init_translation_perturbation,
            perturb_rotation=solver_params.init_rotation_perturbation,
        )
    elif solver_params.init_technique == "random":
        set_pose_init_random(initial_values, data)
    elif solver_params.init_technique == "custom":
        assert (
            solver_params.custom_init_file is not None
        ), "Must provide custom_init_filepath if using custom init"
        custom_vals = load_custom_init_file(solver_params.custom_init_file)
        init_poses = custom_vals.poses
        set_pose_init_custom(initial_values, init_poses)
    else:
        raise ValueError(f"Unknown init technique: {solver_params.init_technique}")

    if solver_params.landmark_init == "custom":
        assert (
            solver_params.custom_init_file is not None
        ), "Must provide custom_init_filepath if using custom init"
        custom_vals = load_custom_init_file(solver_params.custom_init_file)
        init_landmarks = custom_vals.landmarks
        set_landmark_init_custom(initial_values, init_landmarks)
    elif solver_params.landmark_init == "random":
        set_landmark_init_random(initial_values, data)
    elif solver_params.landmark_init == "gt":
        set_landmark_init_gt(initial_values, data)
    else:
        raise ValueError(
            f"Unknown landmark init technique: {solver_params.landmark_init}"
        )

    return initial_values


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
            logger.warning(
                f"Perturbing the first pose by {perturb_magnitude} translation and {perturb_rotation} rotation"
            )
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
            rand_vec = get_random_vector(dim=2, bounds=[x_min, x_max, y_min, y_max])
        elif data.dimension == 3:
            z_min = data.z_min - 50
            z_max = data.z_max + 50
            rand_vec = get_random_vector(
                dim=3, bounds=[x_min, x_max, y_min, y_max, z_min, z_max]
            )

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
            np.array([trans_stddev**2] * 2 + [rot_stddev**2])
        )
    elif data.dimension == 3:
        prior_uncertainty = noiseModel.Diagonal.Sigmas(
            np.array([trans_stddev**2] * 3 + [rot_stddev**2] * 3)
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
        np.array([x_stddev**2, y_stddev**2])
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


def get_factor_graph_from_pyfg_data(data: FactorGraphData) -> NonlinearFactorGraph:
    factor_graph = NonlinearFactorGraph()

    # form objective function
    add_all_costs(factor_graph, data)

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
    result: Values, time: float, data: FactorGraphData, cost: Optional[float] = None
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
    solved_distances: Dict[Tuple[str, str], np.ndarray] = {}
    dim = data.dimension

    poses_have_times = data.all_poses_have_times
    pose_timestamps: Dict[str, float] = {}

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

        solved_distances[assoc] = np.array([np.linalg.norm(trans1 - trans2)])

    for pose_chain in data.pose_variables:
        for pose_var in pose_chain:
            _load_pose_result_to_solved_poses(pose_var)
            if poses_have_times:
                pose_timestamps[pose_var.name] = pose_var.timestamp
            else:
                pose_timestamps[pose_var.name] = get_time_idx_from_frame_name(
                    pose_var.name
                )

    for landmark in data.landmark_variables:
        _load_landmark_result_to_solved_landmarks(landmark)

    for range_measurement in data.range_measurements:
        association = range_measurement.association
        _load_distance_result_to_solved_distances(association)

    return SolverResults(
        VariableValues(
            dim=dim,
            poses=solved_poses,
            landmarks=solved_landmarks,
            distances=solved_distances,
            pose_times=pose_timestamps,
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
