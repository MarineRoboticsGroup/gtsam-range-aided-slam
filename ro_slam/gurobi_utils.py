from __future__ import annotations

import gurobipy as gp  # type: ignore
import numpy as np
import attr
from typing import List, Tuple, Union, Dict

from ro_slam.factor_graph.factor_graph import FactorGraphData


def add_pose_variables(
    model: gp.Model, data: FactorGraphData
) -> Tuple[List[GurobiVarMatrix], List[GurobiVarMatrix]]:
    """
    Adds variables to the model that represent the pose of the robot.
    The variables are added to the model in the order of the pose_variables.

    Args:
        model (gp.Model): The gurobi model to add the variables to.
        pose_variables (List[PoseVariable]): The list of pose variables to add.

    Returns:
        List[GurobiVarMatrix]: The list of variables representing the translations of
        the robot.
    """
    translations: List[GurobiVarMatrix] = []
    rotations: List[GurobiVarMatrix] = []

    for pose_idx, pose in enumerate(data.pose_variables):
        # add new translation variables d-dimensional vector
        new_trans: List[gp.Var] = []
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
        translations.append(GurobiVarMatrix(new_trans, (len(new_trans), 1)))

        # add new rotation variable (dxd rotation matrix)
        new_rot: List[gp.Var] = []
        for i in range(data.dimension):
            for ii in range(data.dimension):
                name = f"rotation_{i}{ii}_p{pose_idx}"
                new_rot.append(add_rotation_var(model, name))

        if data.dimension == 2:
            rotations.append(GurobiVarMatrix(new_rot, (2, 2)))
        elif data.dimension == 3:
            rotations.append(GurobiVarMatrix(new_rot, (3, 3)))

        # add in rotation constraint (must be in orthogonal group)
        I_d = np.eye(data.dimension)
        cnt = 0
        for i in range(data.dimension):
            # this is the i-th column of the rotation matrix
            col_i = rotations[-1].column_as_MVar(i)
            for j in range(i, data.dimension):
                # this is the j-th column of the rotation matrix
                col_j = rotations[-1].column_as_MVar(j)
                model.addConstr(
                    col_i @ col_j == I_d[i, j],
                    name=f"rot_constr_{pose_idx}_{cnt}",
                )
                cnt += 1

    return translations, rotations


def add_landmark_variables(
    model: gp.Model, data: FactorGraphData
) -> List[GurobiVarMatrix]:
    """
    Adds variables to the model that represent the landmarks of the robot.

    Args:
        model (gp.Model): The gurobi model to add the variables to.
        landmarks (List[LandmarkVariable]): The list of landmarks to add.

    Returns:
        List[GurobiVarMatrix]: The list of variables representing the landmarks
        of the robot.
    """
    landmarks = []
    for landmark_idx, landmark in enumerate(data.landmark_variables):
        new_landmark: List[gp.Var] = []
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
        landmarks.append(GurobiVarMatrix(new_landmark, (len(new_landmark), 1)))
    return landmarks


def add_distance_variables(
    model: gp.Model,
    data: FactorGraphData,
    translations: List[GurobiVarMatrix],
    landmarks: List[GurobiVarMatrix],
) -> Dict[Tuple[int, int], gp.Var]:
    """
    Adds variables to the model that represent the distances between the robot's
    landmarks and the landmarks.

    Args:
        model (gp.Model): The gurobi model to add the variables to.
        landmarks (List[LandmarkVariable]): The list of landmarks to add.

    Returns:
        List[gp.Var]: The list of variables representing the distances
        between the robot's landmarks and the landmarks.
    """
    distances = {}
    for dist_idx, range_measure in enumerate(data.range_measurements):
        pose_idx = range_measure.pose_idx
        landmark_idx = range_measure.landmark_idx

        # create distance variable
        dist_key = (pose_idx, landmark_idx)
        distances[dist_key] = model.addVar(name=f"d_p{pose_idx}_l{landmark_idx}")

        # create distance constraint
        # ||t_i - l_j||^2 <= d_ij^2
        trans_i = translations[pose_idx]
        land_j = landmarks[landmark_idx]
        diff = trans_i - land_j
        model.addConstr(
            diff.frob_norm_squared <= distances[dist_key] * distances[dist_key]
        )

    return distances


def get_distances_cost(
    distances: Dict[Tuple[int, int], gp.Var], data: FactorGraphData
) -> gp.QuadExpr:
    """[summary]

    Args:
        distances (Dict[Tuple[int, int], gp.Var]): [description]
        data (FactorGraphData): [description]

    Returns:
        gp.QuadExpr: [description]
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
    translations: List[GurobiVarMatrix],
    rotations: List[GurobiVarMatrix],
    data: FactorGraphData,
) -> gp.QuadExpr:
    """Get the cost associated with the odometry measurements

    Args:
        translations (List[GurobiVarMatrix]): [description]
        rotations (List[GurobiVarMatrix]): [description]
        data (FactorGraphData): [description]

    Returns:
        gp.QuadExpr: [description]
    """
    cost = 0
    for i, odom_measure in enumerate(data.odom_measurements):

        # the indices of the related poses in the odometry measurement
        i_idx = odom_measure.base_pose_idx
        j_idx = odom_measure.to_pose_idx

        # translation component of cost
        # k_ij * ||t_i - t_j - R_i @ t_ij^meas||^2
        trans_weight = odom_measure.translation_weight
        trans_measure = odom_measure.translation_vector
        t_i = translations[i_idx]
        t_j = translations[j_idx]
        R_i = rotations[i_idx]
        term = t_i - t_j - R_i @ trans_measure
        cost += trans_weight * term.frob_norm_squared

        # rotation component of cost
        rot_weight = odom_measure.rotation_weight
        rot_measure = odom_measure.rotation_matrix
        diff_rot_matrix = rotations[j_idx] - rotations[i_idx] @ rot_measure
        cost += rot_weight * diff_rot_matrix.frob_norm_squared
    return cost


def pin_first_pose(translation: GurobiVarMatrix, rotation: GurobiVarMatrix) -> None:
    """
    Pin the first pose of the robot to the origin.

    Args:
        model (gp.Model): The gurobi model to add the variable to.

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


def add_rotation_var(model: gp.Model, name: str) -> gp.Var:
    """
    Adds a variable to the model that represents one component of the rotation
    matrix of the robot.

    Args:
        model (gp.Model): The gurobi model to add the variable to.
        name (str): The name of the variable.

    Returns:
        gp.Var: The variable representing the rotation component of the robot.
    """
    var = model.addVar(lb=-1.0, ub=1.0, name=name)
    return var


def add_translation_var(model: gp.Model, name: str) -> gp.Var:
    """
    Adds a variable to the model that represents one component of the translation
    vector of the robot.

    Args:
        model (gp.Model): The gurobi model to add the variable to.
        name (str): The name of the variable.

    Returns:
        gp.Var: The variable representing the translation component of the robot.
    """
    var = model.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, name=name)
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
    def frob_norm_squared(self) -> gp.QuadExpr:
        """
        Frobenius norm of the matrix
        """
        return gp.quicksum(
            [self.contents[i] * self.contents[i] for i in range(len(self.contents))]
        )

    def __add__(self, other: Matrix) -> GurobiLinExprMatrix:
        """
        Add two matrices together
        """
        res = [self.contents[i] + other.contents[i] for i in range(len(self.contents))]
        return GurobiLinExprMatrix(res, self.shape)

    def __neg__(self) -> Matrix:
        """
        Negation of a matrix
        """
        res = [-self.contents[i] for i in range(len(self.contents))]
        if isinstance(self, GurobiLinExprMatrix):
            return GurobiLinExprMatrix(res, self.shape)
        elif isinstance(self, GurobiVarMatrix):
            return GurobiVarMatrix(res, self.shape)
        else:
            raise ValueError("Unsupported matrix type")

    def __sub__(self, other: Matrix) -> Matrix:
        """
        Subtraction of two matrices
        """
        return self + (-other)


@attr.s(frozen=True)
class GurobiVarMatrix(Matrix):
    """
    A wrapper for a Gurobi matrix, internally represented as a list of gurobi
    variables

    Args:
        mat (List[gp.Var]): the matrix
        shape (Tuple[int, int]): the shape of the matrix
    """

    #: The underlying Gurobi matrix
    contents: List[gp.Var] = attr.ib()
    shape: Tuple[int, int] = attr.ib()

    @property
    def num_rows(self) -> int:
        return self.shape[0]

    @property
    def num_cols(self) -> int:
        return self.shape[1]

    def __add__(self, other: Matrix) -> GurobiLinExprMatrix:
        """Matrix addition"""
        assert self.shape == other.shape, "Matrices are not the same shape"
        diff = []
        for i in range(len(self.contents)):
            diff.append(self.contents[i] + other.contents[i])

        return GurobiLinExprMatrix(diff, self.shape)

    def __neg__(self) -> GurobiLinExprMatrix:
        """Negates the matrix"""
        negated = []
        for i in range(len(self.contents)):
            negated.append(-self.contents[i])
        return GurobiLinExprMatrix(negated, self.shape)

    def __sub__(self, other: Matrix) -> GurobiLinExprMatrix:
        """Subtracts two matrices"""
        assert self.shape == other.shape, "Matrices are not the same shape"
        diff = []
        for i in range(len(self.contents)):
            diff.append(self.contents[i] + (-other.contents[i]))
        return GurobiLinExprMatrix(diff, self.shape)

    def __matmul__(self, other: np.ndarray) -> Union[GurobiLinExprMatrix]:
        """Matrix multiplication"""
        # make sure that dimensions match up
        assert (
            self.shape[1] == other.shape[0]
        ), f"Matrix shapes do not match:{self.shape} != {other.shape}"

        if isinstance(other, np.ndarray):
            mult_result: List[gp.LinExpr] = []

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
                    row_col_product = gp.quicksum(
                        [col_j[k] * row_i[k] for k in range(len(col_j))]
                    )
                    mult_result.append(row_col_product)

            new_shape = (self.shape[0], other.shape[1])
            return GurobiLinExprMatrix(mult_result, new_shape)

        else:
            raise NotImplementedError(
                "Multiplying by non-numpy objects is not supported"
            )

    def col_as_Var_list(self, col_idx: int) -> List[gp.Var]:
        """Returns a vector representing the column of a matrix
        where the matrix is represented as a list of variables
        and the vector is represented as a list of linear expressions

        Args:
        mat (List[gp.Var]): the matrix
        col_idx (int): the index of the column

        Returns:
        List[gp.Var]: the column of the matrix
        """
        num_rows = self.shape[0]
        num_cols = self.shape[1]
        col_i = [self.contents[col_idx + k * num_cols] for k in range(num_rows)]
        return col_i

    def row_as_Var_list(self, row_idx: int) -> List[gp.Var]:
        """Returns a vector representing the row of a matrix
        where the matrix is represented as a list of variables
        and the vector is represented as a list of linear expressions

        Args:
            mat (List[gp.Var]): the matrix
            row_idx (int): the index of the row

        Returns:
            List[gp.Var]: the row of the matrix
        """
        num_rows = self.shape[0]
        num_cols = self.shape[1]
        row_i = [self.contents[k + row_idx * num_cols] for k in range(num_rows)]
        return row_i

    def column_as_MVar(self, col_idx: int) -> gp.MVar:
        """Returns a vector representing the column of a matrix
        where the matrix is represented as a list of variables
        and the vector is represented as a list of linear expressions

        Args:
            mat (List[gp.Var]): the matrix
            col_idx (int): the index of the column

        Returns:
            gp.MVar: the column of the matrix
        """
        col_i = gp.MVar(self.col_as_Var_list(col_idx))
        return col_i

    def row_as_MVar(self, row_idx: int) -> gp.MVar:
        """Returns a vector representing the row of a matrix
        where the matrix is represented as a list of variables
        and the vector is represented as a list of linear expressions

        Args:
            mat (List[gp.Var]): the matrix
            row_idx (int): the index of the row

        Returns:
            gp.MVar: the row of the matrix
        """
        row_i = gp.MVar(self.row_as_Var_list(row_idx))
        return row_i


@attr.s(frozen=True)
class GurobiLinExprMatrix(Matrix):
    """
    A wrapper for a Gurobi matrix, internally represented as a list of gurobi
    linear expressions

    Args:
        contents (List[gp.LinExpr]): the matrix
        shape (Tuple[int, int]): the shape of the matrix
    """

    contents: List[gp.LinExpr] = attr.ib()
    shape: Tuple[int, int] = attr.ib()
