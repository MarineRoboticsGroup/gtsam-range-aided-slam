import numpy as np
import attr
from typing import List, Tuple, Union, Dict
import re

from ro_slam.factor_graph.factor_graph import FactorGraphData
from pydrake.solvers.mathematicalprogram import MathematicalProgram, Solve
from pydrake.solvers.mixed_integer_rotation_constraint import (
    MixedIntegerRotationConstraintGenerator as MIRCGenerator,
)


def add_pose_variables(
    model: MathematicalProgram, data: FactorGraphData
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Adds variables to the model that represent the pose of the robot.
    The variables are added to the model in the order of the pose_variables.

    Args:
        model (MathematicalProgram): The gurobi model to add the variables to.
        pose_variables (List[PoseVariable]): The list of pose variables to add.

    Returns:
        List[np.ndarray]: The list of variables representing the translations of
            the robot.
        List[np.ndarray]: The list of variables representing the rotations of
            the robot.
    """
    translations: List[np.ndarray] = []
    rotations: List[np.ndarray] = []
    rot_const_generator = MIRCGenerator()

    for pose_idx, pose in enumerate(data.pose_variables):
        # add new translation variables d-dimensional vector
        trans_name = f"pose{pose_idx}_translation"
        translations.append(add_translation_var(model, trans_name, data.dimension))

        rot_name = f"pose{pose_idx}_rotation"
        rotations.append(add_rotation_var(model, rot_name, data.dimension))

        # this is one type of constraint that Drake allows, but is a
        # mixed-integer linear constraint so may be more efficient approaches
        rot_const_generator.AddToProgram(rotations[-1], model)

        # TODO test out more efficient constraints?
        # add in rotation constraint (must be in orthogonal group)
        # I_d = np.eye(data.dimension)
        # cnt = 0
        # for i in range(data.dimension):
        #     # this is the i-th column of the rotation matrix
        #     col_i = rotations[-1].column_as_MVar(i)
        #     for j in range(i, data.dimension):
        #         # this is the j-th column of the rotation matrix
        #         col_j = rotations[-1].column_as_MVar(j)
        #         model.Equation(
        #             col_i @ col_j == I_d[i, j],
        #             name=f"rot_constr_{pose_idx}_{cnt}",
        #         )
        #         cnt += 1

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
    socp_relax: bool = False,
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
    for dist_idx, range_measure in enumerate(data.range_measurements):
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
            # TODO not sure if this is correct syntax? Need to test.
            model.AddLorentzConeConstraint(distances[dist_key], diff)
        else:
            # nonconvex quadratic constraint
            model.AddConstraint(
                (diff ** 2).sum() == distances[dist_key] * distances[dist_key]
            )

    return distances


def set_distance_init_gt(
    distances: Dict[Tuple[int, int], np.ndarray], data: FactorGraphData
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
        distances[dist_key].start = range_measure.dist


def init_rotation_variable(rot: np.ndarray, mat: np.ndarray):
    """
    Initialize the rotation variables to the given rotation matrix.

    Args:
        rot (np.ndarray): The rotation variables.
        mat (np.ndarray): The rotation matrix.
    """
    assert rot.shape == mat.shape
    for i in range(mat.shape[0]):
        for ii in range(mat.shape[1]):
            idx = i * mat.shape[1] + ii
            rot.contents[idx].start = mat[i, ii]


def init_translation_variable(trans: np.ndarray, vec: np.ndarray):
    """Initialize the translation variables to the given vector

    Args:
        trans (np.ndarray): the variables to initialize
        vec (np.ndarray): the vector
    """
    vec = vec.reshape(-1, 1)
    assert trans.shape == vec.shape, f"trans shape: {trans.shape} vec shape {vec.shape}"

    for i in range(len(vec)):
        trans.contents[i].start = vec[i]


def set_rotation_init_compose(
    rotations: List[np.ndarray], data: FactorGraphData
) -> None:
    """initializes the rotations by composing the rotations along the odometry chain

    Args:
        rotations (List[np.ndarray]): the rotation variables to initialize
        data (FactorGraphData): the data to use to initialize the rotations
    """
    print("Setting rotation initial points by pose composition")

    # initialize the first rotation to the identity matrix
    curr_pose = np.eye(data.dimension)
    init_rotation_variable(rotations[0], curr_pose)

    # iterate over measurements and init the rotations
    for measure_idx, odom_measure in enumerate(data.odom_measurements):

        # update the current pose
        curr_pose = curr_pose @ odom_measure.rotation_matrix

        # initialize the rotation variables
        cur_gk_rot_variable = rotations[measure_idx + 1]
        init_rotation_variable(cur_gk_rot_variable, curr_pose)


def set_rotation_init_gt(
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
        init_rotation_variable(gk_rotation, true_rotation)


def set_translation_init_gt(
    translations: List[np.ndarray],
    data: FactorGraphData,
) -> None:
    """Initialize the translation and rotation variables to the ground truth translation
    variables.
    """
    print("Setting translation initial points to ground truth")
    for pose_idx, pose in enumerate(data.pose_variables):
        gk_translation = translations[pose_idx]
        true_translation = pose.true_position

        for i in range(data.dimension):
            gk_translation.contents[i].start = pose.true_position[i]


def set_translation_init_compose(
    translations: List[np.ndarray], data: FactorGraphData
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
    init_translation_variable(translations[0], curr_trans)
    print(0, curr_trans)

    for measure_idx, odom_measure in enumerate(data.odom_measurements):

        # update the current pose
        curr_pose = curr_pose @ odom_measure.transformation_matrix
        curr_trans = curr_pose[:-1, -1]
        print(measure_idx, curr_trans)

        # initialize the translation variables
        init_translation_variable(translations[measure_idx + 1], curr_trans)


def set_landmark_init_gt(
    landmarks: List[np.ndarray], data: FactorGraphData, model: MathematicalProgram
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
                model.Equation(
                    gk_landmark_var.contents[i] == true_landmark.true_position[i],
                    name=f"fix_{i}_landmark_{landmark_idx}",
                )
            else:
                gk_landmark_var.contents[i].start = true_pos[i]


def get_distances_cost(
    distances: Dict[Tuple[int, int], np.ndarray], data: FactorGraphData
) -> MathematicalProgram.QuadExpr:
    """[summary]

    Args:
        distances (Dict[Tuple[int, int], np.ndarray]): [description]
        data (FactorGraphData): [description]

    Returns:
        MathematicalProgram.QuadExpr: [description]
    """
    cost = 0
    for range_measure in data.range_measurements:
        pose_idx = range_measure.pose_idx
        landmark_idx = range_measure.landmark_idx

        # create distance variable
        dist_key = (pose_idx, landmark_idx)

        # add in distance cost component
        # k_ij * ||d_ij - d_ij^meas||^2
        dist_diff = distances[dist_key] - range_measure.dist  # distance difference

        cost += dist_diff * dist_diff * range_measure.weight

    return cost


def get_odom_cost(
    translations: List[np.ndarray],
    rotations: List[np.ndarray],
    data: FactorGraphData,
) -> MathematicalProgram.QuadExpr:
    """Get the cost associated with the odometry measurements

    Args:
        translations (List[np.ndarray]): [description]
        rotations (List[np.ndarray]): [description]
        data (FactorGraphData): [description]

    Returns:
        MathematicalProgram.QuadExpr: [description]
    """
    cost = 0
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
        cost += trans_weight * term.frob_norm_squared

        # rotation component of cost
        rot_weight = odom_measure.rotation_weight
        rot_measure = odom_measure.rotation_matrix
        diff_rot_matrix = R_j - (R_i @ rot_measure)
        cost += rot_weight * diff_rot_matrix.frob_norm_squared
    return cost


def pin_first_pose(translation: np.ndarray, rotation: np.ndarray) -> None:
    """
    Pin the first pose of the robot to the origin.

    Args:
        model (MathematicalProgram): The gurobi model to add the variable to.

    """
    # fix translation to origin
    for i in range(translation.shape[0]):
        translation.contents[i].lb = 0.0
        translation.contents[i].ub = 0.0

    # fix rotation to identity
    I_d = np.eye(3)
    for i in range(translation.shape[0]):
        for j in range(translation.shape[0]):
            idx = i * translation.shape[0] + j
            rotation.contents[idx].lb = I_d[i, j]
            rotation.contents[idx].ub = I_d[i, j]


def add_distance_var(model: MathematicalProgram, name: str) -> np.ndarray:
    """
    Add a variable to the model.

    Args:
        model (MathematicalProgram): The model to add the variable to.
        name (str): The name of the variable.

    Returns:
        np.ndarray: The variable.
    """
    return model.addContinuousVariables(1, name=name)


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
    var = model.addContinuousVariables(row=dim, col=dim, name=name)
    return var


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
    var = model.newContinuousVariables(rows=dim, name=name)
    return var
