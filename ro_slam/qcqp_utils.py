import numpy as np
import attr
from typing import List, Tuple, Union, Dict
import re

from ro_slam.factor_graph.factor_graph import FactorGraphData
from ro_slam.utils import _check_square, _check_rotation_matrix

from pydrake.solvers.mathematicalprogram import MathematicalProgram, QuadraticConstraint  # type: ignore
from pydrake.solvers.mixed_integer_rotation_constraint import (  # type: ignore
    MixedIntegerRotationConstraintGenerator as MIRCGenerator,
)

# from pydrake.solvers.mixed_integer_optimization_util import IntervalBinning  # type: ignore

##### Add variables #####


def add_pose_variables(
    model: MathematicalProgram, data: FactorGraphData, orthogonal_constraint: bool
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Adds variables to the model that represent the pose of the robot.
    The variables are added to the model in the order of the pose_variables.

    Args:
        model (MathematicalProgram): The gurobi model to add the variables to.
        pose_variables (List[PoseVariable]): The list of pose variables to add.
        orthogonal_constraint (bool): Whether to add orthogonal constraints to
            the rotation matrices (i.e. R.T @ R = I)

    Returns:
        List[np.ndarray]: The list of variables representing the translations of
            the robot.
        List[np.ndarray]: The list of variables representing the rotations of
            the robot.
    """
    translations: List[np.ndarray] = []
    rotations: List[np.ndarray] = []

    for pose_idx, pose in enumerate(data.pose_variables):
        # add new translation variables d-dimensional vector
        trans_name = f"pose{pose_idx}_translation"
        translations.append(add_translation_var(model, trans_name, data.dimension))

        rot_name = f"pose{pose_idx}_rotation"
        rot_var = add_rotation_var(model, rot_name, data.dimension)
        rotations.append(rot_var)

        # TODO test out more efficient constraints?
        # add in rotation constraint (i.e. matrix must be in orthogonal group)
        if orthogonal_constraint:
            set_orthogonal_constraint(model, rot_var)

    return translations, rotations


def add_landmark_variables(
    model: MathematicalProgram, data: FactorGraphData
) -> List[np.ndarray]:
    """
    Adds variables to the model that represent the landmarks of the robot.

    Args:
        model (MathematicalProgram): The model to add the variables to.
        data (FactorGraphData): The factor graph data.

    Returns:
        List[np.ndarray]: The list of variables representing the landmarks
        of the robot.
    """
    landmarks = []
    for landmark_idx, landmark in enumerate(data.landmark_variables):
        name = f"landmark{landmark_idx}_translation"
        landmarks.append(add_translation_var(model, name, data.dimension))
    return landmarks


def add_distance_variables(
    model: MathematicalProgram,
    data: FactorGraphData,
    translations: List[np.ndarray],
    landmarks: List[np.ndarray],
    socp_relax: bool,
) -> Dict[Tuple[int, int], np.ndarray]:
    """
    Adds variables to the model that represent the distances between the robot's
    landmarks and the landmarks.

    Args:
        model (MathematicalProgram): The gurobi model to add the variables to.
        landmarks (List[LandmarkVariable]): The list of landmarks to add.

    Returns:
        Dict[Tuple[int, int], np.ndarray]: The dict of variables representing
        the distances between the robot's landmarks and the landmarks.
    """
    distances = {}
    for range_measure in data.range_measurements:
        pose_idx = range_measure.pose_idx
        landmark_idx = range_measure.landmark_idx

        # create distance variable
        dist_key = (pose_idx, landmark_idx)
        # name = f"d_p{pose_"
        name = f"d_p{pose_idx}_l{landmark_idx}"
        distances[dist_key] = add_distance_var(model, name)

        # create distance constraint
        # ||t_i - l_j||^2 <= d_ij^2
        trans_i = translations[pose_idx]
        land_j = landmarks[landmark_idx]
        diff = trans_i - land_j

        # give two options for how to implement the distance constraint
        if socp_relax:
            cone_vars = np.asarray([distances[dist_key][0], *diff.flatten()])
            cone_const = model.AddLorentzConeConstraint(cone_vars)
        else:
            # nonconvex quadratic constraint
            add_drake_distance_equality_constraint(
                model, trans_i, land_j, distances[dist_key]
            )

    return distances


def add_distance_var(model: MathematicalProgram, name: str) -> np.ndarray:
    """
    Add a variable to the model.

    Args:
        model (MathematicalProgram): The model to add the variable to.
        name (str): The name of the variable.

    Returns:
        np.ndarray: The variable.
    """
    dist_var = model.NewContinuousVariables(1, name=name)
    model.AddBoundingBoxConstraint(0, np.inf, dist_var)
    return dist_var


def add_rotation_var(model: MathematicalProgram, name: str, dim: int) -> np.ndarray:
    """
    Adds a variable to the model that represents the rotation
    matrix of a single pose.

    Args:
        model (MathematicalProgram): The model to add the variable to.
        name (str): The name of the variable.
        dim (int): The dimension of the ambient space

    Returns:
        np.ndarray: The variable representing the rotation of the robot
    """
    rot_var = model.NewContinuousVariables(rows=dim, cols=dim, name=name)
    model.AddBoundingBoxConstraint(-1, 1, rot_var)
    return rot_var


def add_translation_var(model: MathematicalProgram, name: str, dim: int) -> np.ndarray:
    """
    Adds a variable to the model that represents a translation vector.

    Args:
        model (MathematicalProgram): The gurobi model to add the variable to.
        name (str): The name of the variable.
        dim (int): The dimension of the translation vector.

    Returns:
        np.ndarray: The variable representing the translation component of the robot.
    """
    assert dim == 2 or dim == 3
    var = model.NewContinuousVariables(rows=dim, name=name)
    model.AddBoundingBoxConstraint(-np.inf, np.inf, var)
    return var


##### Add costs #####


def add_distances_cost(
    model: MathematicalProgram,
    distances: Dict[Tuple[int, int], np.ndarray],
    data: FactorGraphData,
):
    """Adds in the cost due to the distances as:
    sum_{i,j} k_ij * ||d_ij - d_ij^meas||^2

    Args:
        model (MathematicalProgram): the model to add the cost to
        distances (Dict[Tuple[int, int], np.ndarray]): [description]
        data (FactorGraphData): [description]

    """
    for range_measure in data.range_measurements:
        pose_idx = range_measure.pose_idx
        landmark_idx = range_measure.landmark_idx

        # create distance variable
        dist_key = (pose_idx, landmark_idx)

        # add in distance cost component
        # k_ij * ||d_ij - d_ij^meas||^2
        dist_diff = distances[dist_key] - range_measure.dist

        model.AddQuadraticCost(range_measure.weight * (dist_diff ** 2).sum())


def add_odom_cost(
    model: MathematicalProgram,
    translations: List[np.ndarray],
    rotations: List[np.ndarray],
    data: FactorGraphData,
):
    """Add the cost associated with the odometry measurements as:

        translation component of cost
        k_ij * ||t_i - t_j - R_i @ t_ij^meas||^2

        rotation component of cost
        tau_ij * || R_j - (R_i @ R_ij^\top) ||_\frob^2

    Args:
        model (MathematicalProgram): the model to add the cost to
        translations (List[np.ndarray]): [description]
        rotations (List[np.ndarray]): [description]
        data (FactorGraphData): [description]

    """
    for odom_measure in data.odom_measurements:

        # the indices of the related poses in the odometry measurement
        i_idx = odom_measure.base_pose_idx
        j_idx = odom_measure.to_pose_idx

        # get the translation and rotation variables
        t_i = translations[i_idx]
        t_j = translations[j_idx]
        R_i = rotations[i_idx]
        R_j = rotations[j_idx]

        # translation component of cost
        # k_ij * ||t_i - t_j - R_i @ t_ij^meas||^2
        trans_weight = odom_measure.translation_weight
        trans_measure = odom_measure.translation_vector
        term = t_j - t_i - (R_i @ trans_measure)
        model.AddQuadraticCost(trans_weight * (term ** 2).sum(), is_convex=True)

        # rotation component of cost
        # tau_ij * || R_j - (R_i @ R_ij^\top) ||_\frob
        rot_weight = odom_measure.rotation_weight
        rot_measure = odom_measure.rotation_matrix
        diff_rot_matrix = R_j - (R_i @ rot_measure)
        model.AddQuadraticCost(
            rot_weight * (diff_rot_matrix ** 2).sum(), is_convex=True
        )


##### Initialization strategies #####


def set_distance_init_gt(
    model: MathematicalProgram,
    distances: Dict[Tuple[int, int], np.ndarray],
    data: FactorGraphData,
):
    """Initialize the distance variables to the ground truth distances.

    Args:
        distances (Dict[Tuple[int, int], np.ndarray]): [description]
        data (FactorGraphData): [description]
    """
    print("Setting distance initial points to measured distance")
    for range_measure in data.range_measurements:
        pose_idx = range_measure.pose_idx
        landmark_idx = range_measure.landmark_idx
        dist_key = (pose_idx, landmark_idx)
        model.SetInitialGuess(distances[dist_key][0], range_measure.dist)


def init_rotation_variable(
    model: MathematicalProgram, rot: np.ndarray, mat: np.ndarray
):
    """
    Initialize the rotation variables to the given rotation matrix.

    Args:
        rot (np.ndarray): The rotation variables.
        mat (np.ndarray): The rotation matrix.
    """
    assert rot.shape == mat.shape
    model.SetInitialGuess(rot, mat)


def init_translation_variable(
    model: MathematicalProgram, trans: np.ndarray, vec: np.ndarray
):
    """Initialize the translation variables to the given vector

    Args:
        trans (np.ndarray): the variables to initialize
        vec (np.ndarray): the vector
    """
    assert trans.shape == vec.shape, f"trans shape: {trans.shape} vec shape {vec.shape}"
    model.SetInitialGuess(trans, vec)


def set_rotation_init_compose(
    model: MathematicalProgram, rotations: List[np.ndarray], data: FactorGraphData
) -> None:
    """initializes the rotations by composing the rotations along the odometry chain

    Args:
        rotations (List[np.ndarray]): the rotation variables to initialize
        data (FactorGraphData): the data to use to initialize the rotations
    """
    print("Setting rotation initial points by pose composition")

    # initialize the first rotation to the identity matrix
    curr_pose = np.eye(data.dimension)
    init_rotation_variable(model, rotations[0], curr_pose)

    # iterate over measurements and init the rotations
    for measure_idx, odom_measure in enumerate(data.odom_measurements):

        # update the current pose
        curr_pose = curr_pose @ odom_measure.rotation_matrix

        # initialize the rotation variables
        cur_gk_rot_variable = rotations[measure_idx + 1]
        init_rotation_variable(model, cur_gk_rot_variable, curr_pose)


def set_rotation_init_gt(
    model: MathematicalProgram,
    rotations: List[np.ndarray],
    data: FactorGraphData,
) -> None:
    """Initialize the translation and rotation variables to the ground truth translation
    variables.
    """
    print("Setting rotation initial points to ground truth")
    for pose_idx, pose in enumerate(data.pose_variables):
        gk_rotation = rotations[pose_idx]
        true_rotation = pose.rotation_matrix
        init_rotation_variable(model, gk_rotation, true_rotation)


def set_translation_init_gt(
    model: MathematicalProgram,
    translations: List[np.ndarray],
    data: FactorGraphData,
) -> None:
    """Initialize the translation and rotation variables to the ground truth translation
    variables.
    """
    print("Setting translation initial points to ground truth")
    for pose_idx, pose in enumerate(data.pose_variables):
        translation_var = translations[pose_idx]
        true_translation = pose.true_position
        init_translation_variable(model, translation_var, np.asarray(true_translation))


def set_translation_init_compose(
    model: MathematicalProgram, translations: List[np.ndarray], data: FactorGraphData
) -> None:
    """Initialize the translation variables by composing the translation
    variables along the odometry chain.

    Args:
        translations (List[np.ndarray]): the translation variables to
            initialize
        data (FactorGraphData): the data to use to initialize the translation
    """
    print("Setting translation initial points by pose composition")
    curr_pose = np.eye(data.dimension + 1)
    curr_trans = curr_pose[:-1, -1]
    init_translation_variable(model, translations[0], curr_trans)

    for measure_idx, odom_measure in enumerate(data.odom_measurements):

        # update the current pose
        curr_pose = curr_pose @ odom_measure.transformation_matrix
        curr_trans = curr_pose[:-1, -1]

        # initialize the translation variables
        init_translation_variable(model, translations[measure_idx + 1], curr_trans)


def set_landmark_init_gt(
    model: MathematicalProgram,
    landmarks: List[np.ndarray],
    data: FactorGraphData,
):
    """Initialize the landmark variables to the ground truth landmarks.

    Args:
        landmarks (List[np.ndarray]): [description]
        data (FactorGraphData): [description]
    """
    print("Setting landmark initial points to ground truth")
    for landmark_idx, true_landmark in enumerate(data.landmark_variables):
        gk_landmark_var = landmarks[landmark_idx]
        true_pos = true_landmark.true_position
        for i in range(data.dimension):
            if landmark_idx == 0 or False:
                # pin the first landmark to the correct position
                add_drake_matrix_equality_constraint(
                    model, gk_landmark_var, np.asarray(true_pos)
                )
            else:
                # initialize landmark to correct position
                model.SetInitialGuess(gk_landmark_var, true_pos)


##### Constraints #####


def pin_first_pose(
    model: MathematicalProgram, translation: np.ndarray, rotation: np.ndarray
) -> None:
    """
    Pin the first pose of the robot to the origin.

    Args:
        model (MathematicalProgram): The model to pin the pose in
        translation (np.ndarray): The translation variable to pin
        rotation (np.ndarray): The rotation variable to pin

    """
    _check_square(rotation)
    assert len(translation) == rotation.shape[0]

    # fix translation to origin
    add_drake_matrix_equality_constraint(
        model, translation, np.zeros(translation.shape)
    )

    # fix rotation to identity
    d = rotation.shape[0]
    I_d = np.eye(d)
    add_drake_matrix_equality_constraint(model, rotation, I_d)


def set_orthogonal_constraint(model: MathematicalProgram, mat: np.ndarray) -> None:
    """Sets an orthogonal constraint on a given matrix (i.e. R.T @ R == I)

    Args:
        model (MathematicalProgram): the model to set the constraint in
        mat (np.ndarray): the matrix to constrain
    """
    assert mat.shape[0] == mat.shape[1], "matrix must be square"
    assert mat.shape[0] == 2, "only support 2d matrices right now"
    d = mat.shape[0]

    for i in range(d):
        for j in range(i, d):
            col_i = mat[:, i]
            if i == j:
                # set diagonal constraint
                const = model.AddConstraint(col_i[0] ** 2 + col_i[1] ** 2 == 1)
            else:
                # set off-diagonal constraint
                col_j = mat[:, j]
                const = model.AddConstraint(
                    col_i[0] * col_j[0] + col_i[1] * col_j[1] == 0
                )


def set_mixed_int_rotation_constraint(
    model: MathematicalProgram, vars: List[np.ndarray]
):
    """Uses built-in Drake constraints to write the rotation matrix as
    mixed-integer constraints

    Args:
        model (MathematicalProgram): the model to set the constraints in
        vars (List[np.ndarray]): the variables to constrain to rotation matrices
    """
    # * various constraints for rotation matrices
    approach = MIRCGenerator.Approach.kBoxSphereIntersection
    # approach = MIRCGenerator.Approach.kBilinearMcCormick
    # approach = MIRCGenerator.Approach.kBoth

    intervals_per_half_axis = 10

    # binning = MIRCGenerator.Binning.kLinear
    binning = MIRCGenerator.Binning.kLogarithmic

    rot_const_generator = MIRCGenerator(approach, intervals_per_half_axis, binning)

    for mat in vars:
        # this is one type of constraint that Drake allows, but is a
        # mixed-integer linear constraint so may be more efficient approaches
        rot_const_generator.AddToProgram(mat, model)


def add_drake_matrix_equality_constraint(
    model: MathematicalProgram, var: np.ndarray, mat: np.ndarray
) -> None:
    """Adds a Drake matrix equality constraint to the model.

    Args:
        model (MathematicalProgram): the model to add the constraint to
        var (np.ndarray): the variable to constrain
        mat (np.ndarray): the matrix to set it equal to
    """
    assert var.shape == mat.shape, "variable and matrix must have same shape"

    var_vec = var.flatten()
    mat_vec = mat.flatten()
    I = np.eye(len(var_vec))
    model.AddLinearEqualityConstraint(
        I,
        mat_vec,
        var_vec,
    )


def add_drake_distance_equality_constraint(
    model: MathematicalProgram, trans: np.ndarray, land: np.ndarray, dist: np.ndarray
) -> None:
    """Adds a constraint of the form
    ||trans-land||_2^2 == dist**2

    Args:
        model (MathematicalProgram): [description]
        trans (np.ndarray): [description]
        land (np.ndarray): [description]
        dist (np.ndarray): [description]
    """
    assert len(trans) == len(land), "must be same dimension"
    assert len(dist) == 1, "must be a scalar"

    d = len(trans)
    I_d = np.eye(len(trans))
    q_size = 2 * d + 1
    Q = np.zeros((q_size, q_size))
    Q[:2, :2] = I_d
    Q[2:4, 2:4] = I_d
    Q[:2, 2:4] = -I_d
    Q[2:4, :2] = -I_d
    Q[-1, -1] = -1

    quad_constraint = QuadraticConstraint(Q, np.zeros((q_size, 1)), lb=0.0, ub=0.0)
    model.AddConstraint(
        quad_constraint,
        vars=np.asarray([*trans.flatten(), *land.flatten(), dist[0]]).flatten(),
    )
