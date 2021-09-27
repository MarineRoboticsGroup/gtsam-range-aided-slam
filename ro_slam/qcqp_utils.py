from __future__ import annotations

from gekko import GEKKO as gk  # type: ignore
import numpy as np
import attr
from typing import List, Tuple, Union, Dict
import re

from ro_slam.factor_graph.factor_graph import FactorGraphData


def add_pose_variables(
    model: gk, data: FactorGraphData
) -> Tuple[List[VarMatrix], List[VarMatrix]]:
    """
    Adds variables to the model that represent the pose of the robot.
    The variables are added to the model in the order of the pose_variables.

    Args:
        model (gk): The gurobi model to add the variables to.
        pose_variables (List[PoseVariable]): The list of pose variables to add.

    Returns:
        List[VarMatrix]: The list of variables representing the translations of
        the robot.
    """
    translations: List[VarMatrix] = []
    rotations: List[VarMatrix] = []

    for pose_idx, pose in enumerate(data.pose_variables):
        # add new translation variables d-dimensional vector
        new_trans: List[gk.Var] = []
        for i in range(data.dimension):
            if i == 0:
                name = f"translation_x_p{pose_idx}"
            elif i == 1:
                name = f"translation_y_p{pose_idx}"
            elif i == 2:
                name = f"translation_z_p{pose_idx}"
            else:
                raise NotImplementedError

            new_trans.append(add_translation_var(model, name))

        assert len(new_trans) == 2 or len(new_trans) == 3
        translations.append(VarMatrix(new_trans, (len(new_trans), 1)))

        # add new rotation variable (dxd rotation matrix)
        new_rot: List[gk.Var] = []
        for i in range(data.dimension):
            for ii in range(data.dimension):
                name = f"rotation_{i}{ii}_p{pose_idx}"
                new_rot.append(add_rotation_var(model, name))

        if data.dimension == 2:
            rotations.append(VarMatrix(new_rot, (2, 2)))
        elif data.dimension == 3:
            rotations.append(VarMatrix(new_rot, (3, 3)))

        # add in rotation constraint (must be in orthogonal group)
        I_d = np.eye(data.dimension)
        cnt = 0
        for i in range(data.dimension):
            # this is the i-th column of the rotation matrix
            col_i = rotations[-1].column_as_MVar(i)
            for j in range(i, data.dimension):
                # this is the j-th column of the rotation matrix
                col_j = rotations[-1].column_as_MVar(j)
                model.Equation(
                    col_i @ col_j == I_d[i, j],
                    name=f"rot_constr_{pose_idx}_{cnt}",
                )
                cnt += 1

    return translations, rotations


def add_landmark_variables(model: gk, data: FactorGraphData) -> List[VarMatrix]:
    """
    Adds variables to the model that represent the landmarks of the robot.

    Args:
        model (gk): The gurobi model to add the variables to.
        landmarks (List[LandmarkVariable]): The list of landmarks to add.

    Returns:
        List[VarMatrix]: The list of variables representing the landmarks
        of the robot.
    """
    landmarks = []
    for landmark_idx, landmark in enumerate(data.landmark_variables):
        new_landmark: List[gk.Var] = []
        for i in range(data.dimension):
            if i == 0:
                name = f"landmark_x_l{landmark_idx}"
            elif i == 1:
                name = f"landmark_y_l{landmark_idx}"
            elif i == 2:
                name = f"landmark_z_l{landmark_idx}"
            else:
                raise NotImplementedError
            new_landmark.append(add_translation_var(model, name))
        landmarks.append(VarMatrix(new_landmark, (len(new_landmark), 1)))
    return landmarks


def add_distance_variables(
    model: gk,
    data: FactorGraphData,
    translations: List[VarMatrix],
    landmarks: List[VarMatrix],
) -> Dict[Tuple[int, int], gk.Var]:
    """
    Adds variables to the model that represent the distances between the robot's
    landmarks and the landmarks.

    Args:
        model (gk): The gurobi model to add the variables to.
        landmarks (List[LandmarkVariable]): The list of landmarks to add.

    Returns:
        List[gk.Var]: The list of variables representing the distances
        between the robot's landmarks and the landmarks.
    """
    distances = {}
    for dist_idx, range_measure in enumerate(data.range_measurements):
        pose_idx = range_measure.pose_idx
        landmark_idx = range_measure.landmark_idx

        # create distance variable
        dist_key = (pose_idx, landmark_idx)
        distances[dist_key] = model.Var(name=f"d_p{pose_idx}_l{landmark_idx}")

        # create distance constraint
        # ||t_i - l_j||^2 <= d_ij^2
        trans_i = translations[pose_idx]
        land_j = landmarks[landmark_idx]
        diff = trans_i - land_j

        model.Equation(
            diff.frob_norm_squared == distances[dist_key] * distances[dist_key]
        )

    return distances


def set_distance_init_gt(
    distances: Dict[Tuple[int, int], gk.Var], data: FactorGraphData
):
    """Initialize the distance variables to the ground truth distances.

    Args:
        distances (Dict[Tuple[int, int], gk.Var]): [description]
        data (FactorGraphData): [description]
    """
    print("Setting distance initial points to measured distance")
    for range_measure in data.range_measurements:
        pose_idx = range_measure.pose_idx
        landmark_idx = range_measure.landmark_idx
        dist_key = (pose_idx, landmark_idx)
        distances[dist_key].start = range_measure.dist


def init_rotation_variable(rot: VarMatrix, mat: np.ndarray):
    """
    Initialize the rotation variables to the given rotation matrix.

    Args:
        rot (VarMatrix): The rotation variables.
        mat (np.ndarray): The rotation matrix.
    """
    assert rot.shape == mat.shape
    for i in range(mat.shape[0]):
        for ii in range(mat.shape[1]):
            idx = i * mat.shape[1] + ii
            rot.contents[idx].start = mat[i, ii]


def init_translation_variable(trans: VarMatrix, vec: np.ndarray):
    """Initialize the translation variables to the given vector

    Args:
        trans (VarMatrix): the variables to initialize
        vec (np.ndarray): the vector
    """
    vec = vec.reshape(-1, 1)
    assert trans.shape == vec.shape, f"trans shape: {trans.shape} vec shape {vec.shape}"

    for i in range(len(vec)):
        trans.contents[i].start = vec[i]


def set_rotation_init_compose(
    rotations: List[VarMatrix], data: FactorGraphData
) -> None:
    """initializes the rotations by composing the rotations along the odometry chain

    Args:
        rotations (List[VarMatrix]): the rotation variables to initialize
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
    rotations: List[VarMatrix],
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
    translations: List[VarMatrix],
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
    translations: List[VarMatrix], data: FactorGraphData
) -> None:
    """Initialize the translation variables by composing the translation
    variables along the odometry chain.

    Args:
        translations (List[VarMatrix]): the translation variables to
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


def set_landmark_init_gt(landmarks: List[VarMatrix], data: FactorGraphData, model: gk):
    """Initialize the landmark variables to the ground truth landmarks.

    Args:
        landmarks (List[VarMatrix]): [description]
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
    distances: Dict[Tuple[int, int], gk.Var], data: FactorGraphData
) -> gk.QuadExpr:
    """[summary]

    Args:
        distances (Dict[Tuple[int, int], gk.Var]): [description]
        data (FactorGraphData): [description]

    Returns:
        gk.QuadExpr: [description]
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
    translations: List[VarMatrix],
    rotations: List[VarMatrix],
    data: FactorGraphData,
) -> gk.QuadExpr:
    """Get the cost associated with the odometry measurements

    Args:
        translations (List[VarMatrix]): [description]
        rotations (List[VarMatrix]): [description]
        data (FactorGraphData): [description]

    Returns:
        gk.QuadExpr: [description]
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


def pin_first_pose(translation: VarMatrix, rotation: VarMatrix) -> None:
    """
    Pin the first pose of the robot to the origin.

    Args:
        model (gk): The gurobi model to add the variable to.

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


def add_rotation_var(model: gk, name: str) -> gk.Var:
    """
    Adds a variable to the model that represents one component of the rotation
    matrix of the robot.

    Args:
        model (gk): The gurobi model to add the variable to.
        name (str): The name of the variable.

    Returns:
        gk.Var: The variable representing the rotation component of the robot.
    """
    var = model.Var(lb=-1.0, ub=1.0, name=name)
    return var


def add_translation_var(model: gk, name: str) -> gk.Var:
    """
    Adds a variable to the model that represents one component of the translation
    vector of the robot.

    Args:
        model (gk): The gurobi model to add the variable to.
        name (str): The name of the variable.

    Returns:
        gk.Var: The variable representing the translation component of the robot.
    """
    var = model.Var(lb=-gk.GRB.INFINITY, ub=gk.GRB.INFINITY, name=name)
    return var


@attr.s
class Matrix:
    """
    An abstract matrix class to be extended by other wrappers for Gurobi
    matrices
    """

    contents: List = attr.ib()
    shape: Tuple[int, int] = attr.ib()

    @property
    def frob_norm_squared(self) -> gk.QuadExpr:
        """
        Frobenius norm of the matrix
        """
        return gk.quicksum(
            [self.contents[i] * self.contents[i] for i in range(len(self.contents))]
        )

    def __add__(self, other: Matrix) -> EquationMatrix:
        """
        Add two matrices together
        """
        res = [self.contents[i] + other.contents[i] for i in range(len(self.contents))]
        return EquationMatrix(res, self.shape)

    def __neg__(self) -> Matrix:
        """
        Negation of a matrix
        """
        res = [-self.contents[i] for i in range(len(self.contents))]
        if isinstance(self, EquationMatrix):
            return EquationMatrix(res, self.shape)
        elif isinstance(self, VarMatrix):
            return VarMatrix(res, self.shape)
        else:
            raise ValueError("Unsupported matrix type")

    def __sub__(self, other: Matrix) -> Matrix:
        """
        Subtraction of two matrices
        """
        return self + (-other)


@attr.s(frozen=True)
class VarMatrix(Matrix):
    """
    A wrapper for a matrix, internally represented as a list of variables

    Args:
        mat (List[gk.Var]): the matrix
        shape (Tuple[int, int]): the shape of the matrix
    """

    #: The underlying matrix
    contents: List[gk.Var] = attr.ib()
    shape: Tuple[int, int] = attr.ib()

    @property
    def num_rows(self) -> int:
        return self.shape[0]

    @property
    def num_cols(self) -> int:
        return self.shape[1]

    # TODO explain this string
    def __str__(self) -> str:
        var_name = self.contents[0].VarName
        search = re.search(r"_[lp][0-9]*", var_name)
        line = search.group(0)[1:] + " "  # type: ignore
        for i, entry in enumerate(self.contents):
            if i % self.num_cols == 0 and self.num_cols > 1:
                line += "\n"
            line += f"{entry.x:.2f} "

        return line

    def __add__(self, other: Matrix) -> EquationMatrix:
        """Matrix addition"""
        assert self.shape == other.shape, "Matrices are not the same shape"
        diff = []
        for i in range(len(self.contents)):
            diff.append(self.contents[i] + other.contents[i])

        return EquationMatrix(diff, self.shape)

    def __neg__(self) -> EquationMatrix:
        """Negates the matrix"""
        negated = []
        for i in range(len(self.contents)):
            negated.append(-self.contents[i])
        return EquationMatrix(negated, self.shape)

    def __sub__(self, other: Matrix) -> EquationMatrix:
        """Subtracts two matrices"""
        assert self.shape == other.shape, "Matrices are not the same shape"
        diff = []
        for i in range(len(self.contents)):
            diff.append(self.contents[i] + (-other.contents[i]))
        return EquationMatrix(diff, self.shape)

    def __matmul__(self, other: np.ndarray) -> Union[EquationMatrix]:
        """Matrix multiplication"""
        # make sure that dimensions match up
        assert (
            self.shape[1] == other.shape[0]
        ), f"Matrix shapes do not match:{self.shape} != {other.shape}"

        if isinstance(other, np.ndarray):
            mult_result: List[gk.LinExpr] = []

            # remake other into 2-d array if currently just a vector
            if len(other.shape) == 1:
                other = other.reshape(-1, 1)

            # get the number of columns of the other object
            other_num_cols = other.shape[1]

            # perform the multiplication
            for row_idx in range(self.num_rows):
                row_i = self.row_as_Var_list(row_idx)
                for col_idx in range(other_num_cols):
                    col_j = other[:, col_idx]
                    row_col_product = gk.quicksum(
                        [col_j[k] * row_i[k] for k in range(len(col_j))]
                    )
                    mult_result.append(row_col_product)

            new_shape = (self.shape[0], other.shape[1])
            return EquationMatrix(mult_result, new_shape)

        else:
            raise NotImplementedError(
                "Multiplying by non-numpy objects is not supported"
            )

    def col_as_Var_list(self, col_idx: int) -> List[gk.Var]:
        """Returns a vector representing the column of a matrix
        where the matrix is represented as a list of variables
        and the vector is represented as a list of linear expressions

        Args:
        mat (List[gk.Var]): the matrix
        col_idx (int): the index of the column

        Returns:
        List[gk.Var]: the column of the matrix
        """
        num_rows = self.shape[0]
        num_cols = self.shape[1]
        col_i = [self.contents[col_idx + k * num_cols] for k in range(num_rows)]
        return col_i

    def row_as_Var_list(self, row_idx: int) -> List[gk.Var]:
        """Returns a vector representing the row of a matrix
        where the matrix is represented as a list of variables
        and the vector is represented as a list of linear expressions

        Args:
            mat (List[gk.Var]): the matrix
            row_idx (int): the index of the row

        Returns:
            List[gk.Var]: the row of the matrix
        """
        num_rows = self.shape[0]
        num_cols = self.shape[1]
        row_i = [self.contents[k + row_idx * num_cols] for k in range(num_rows)]
        return row_i

    def column_as_MVar(self, col_idx: int) -> gk.MVar:
        """Returns a vector representing the column of a matrix
        where the matrix is represented as a list of variables
        and the vector is represented as a list of linear expressions

        Args:
            mat (List[gk.Var]): the matrix
            col_idx (int): the index of the column

        Returns:
            gk.MVar: the column of the matrix
        """
        col_i = gk.MVar(self.col_as_Var_list(col_idx))
        return col_i

    def row_as_MVar(self, row_idx: int) -> gk.MVar:
        """Returns a vector representing the row of a matrix
        where the matrix is represented as a list of variables
        and the vector is represented as a list of linear expressions

        Args:
            mat (List[gk.Var]): the matrix
            row_idx (int): the index of the row

        Returns:
            gk.MVar: the row of the matrix
        """
        row_i = gk.MVar(self.row_as_Var_list(row_idx))
        return row_i


@attr.s(frozen=True)
class EquationMatrix(Matrix):
    """
    A wrapper for a Gurobi matrix, internally represented as a list of gurobi
    linear expressions

    Args:
        contents (List[gk.LinExpr]): the matrix
        shape (Tuple[int, int]): the shape of the matrix
    """

    contents: List[gk.LinExpr] = attr.ib()
    shape: Tuple[int, int] = attr.ib()
