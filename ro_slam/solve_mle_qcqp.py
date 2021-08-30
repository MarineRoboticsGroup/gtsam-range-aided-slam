import numpy as np
from os.path import expanduser, join
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt  # type: ignore
import scipy.linalg as la  # type: ignore
import gurobipy as gp  # type: ignore
from gurobipy import GRB  # type: ignore

from ro_slam.gurobi_utils import (
    GurobiVarMatrix,
    add_rotation_var,
    add_translation_var,
    pin_first_pose,
    add_pose_variables,
    add_landmark_variables,
    add_distance_variables,
    get_distances_cost,
    get_odom_cost,
)
from ro_slam.factor_graph.parse_factor_graph import parse_factor_graph_file
from ro_slam.factor_graph.factor_graph import (
    OdomMeasurement,
    RangeMeasurement,
    PoseVariable,
    LandmarkVariable,
    PosePrior,
    LandmarkPrior,
    FactorGraphData,
)
from ro_slam.utils import (
    _print_eigvals,
    _check_is_laplacian,
    _check_symmetric,
    _check_psd,
    _general_kron,
    _matprint_block,
)


def _get_range_constraint_matrices(
    data: FactorGraphData,
) -> List[np.ndarray]:
    """gets the range constraint matrix for the lagrangian dual

    args:
        data (FactorGraphData): the data

    returns:
        List[np.ndarray]: the range constraint matrices
    """
    mat_dim = data.poses_and_landmarks_dimension + data.distance_variables_dimension
    d = data.dimension
    constraint_matrices: List[np.ndarray] = []

    # iterate over all range measurements
    for measure in data.range_measurements:
        assert isinstance(measure, RangeMeasurement)

        # get the indices of the corresponding pose translation
        measured_pose = data.get_range_measurement_pose(measure)
        i_start, i_end = data.get_pose_translation_variable_indices(measured_pose)
        assert isinstance(i_start, int) and isinstance(i_end, int)
        assert i_start < i_end
        assert i_start >= 0
        assert i_end <= data.num_poses * d

        # get the indices of the corresponding landmark translation
        measured_landmark = data.get_range_measurement_landmark(measure)
        j_start, j_end = data.get_landmark_translation_variable_indices(
            measured_landmark
        )
        assert isinstance(j_start, int) and isinstance(
            j_end, int
        ), f"bad indices:{j_start}, {j_end}"
        assert j_start < j_end, f"bad indices:{j_start}, {j_end}"
        assert j_start >= data.num_poses * d
        assert (
            j_end <= data.num_translations * d
        ), f"{j_end} > {data.num_translations * d}"

        # this matrix represents the quadratic constraint between the
        # distance variable d_ij and the two translations it is relating
        # (t_i, t_j) -> d_ij^2 = ||t_i - t_j||^2
        Kij = np.zeros((mat_dim, mat_dim))

        # The block matrices on the diagonals corresponding to the
        # translations are identity matrices and the ones on the
        # off-diagonals are negative identity matrices
        Kij[i_start:i_end, i_start:i_end] = np.eye(d)
        Kij[j_start:j_end, j_start:j_end] = np.eye(d)

        Kij[i_start:i_end, j_start:j_end] = -np.eye(d)
        Kij[j_start:j_end, i_start:i_end] = -np.eye(d)

        # add in the component corresponding to the distance variable
        dij_idx = data.get_range_dist_variable_indices(measure)
        Kij[dij_idx, dij_idx] = -1

        # sanity check this
        _check_symmetric(Kij)

        # Finally, we scale this by this constraints corresponding lagrange
        # multiplier and add it all together to the big matrix where we are
        # summing these constraints
        constraint_matrices.append(Kij)

    return constraint_matrices


def _get_rotation_constraint_matrices(
    data: FactorGraphData,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """gets the rotation constraint matrices for the qcqp
    enforcing that R.T @ R = I for all rotations

    args:
        data (FactorGraphData): the data

    returns:
        List[np.ndarray]: the rotation constraint matrices which under quadratic
            product equal 1
        List[np.ndarray]: the rotation constraint matrices which under quadratic
            product equal 0
    """

    mat_dim = data.poses_and_landmarks_dimension + data.distance_variables_dimension
    d = data.dimension
    ones_constraints: List[np.ndarray] = []
    zeros_constraints: List[np.ndarray] = []

    # add lagrange multipliers
    # for _ in range(data.num_poses):
    for pose in data.pose_variables:
        start_idx, end_idx = data.get_pose_rotation_variable_indices(pose)

        # step by 'd' every time to access the 'd' x 'd' subblocks
        # we are adding the quadratic constraints that equal one here
        for i in range(start_idx, end_idx, d):
            E_iuv_ones = np.zeros((mat_dim, mat_dim))
            E_iuv_ones[i : i + d, i : i + d] = np.eye(d)
            ones_constraints.append(E_iuv_ones)

        # step by 'd' every time to access the 'd'x'd' subblocks
        # we are adding the quadratic constraints that equal zero here
        for i in range(start_idx, end_idx, d):
            for j in range(start_idx, end_idx, d):
                if i == j:
                    continue  # skip the diagonal

                assert i != j  # this should never happen
                E_iuv_zeros = np.zeros((mat_dim, mat_dim))
                E_iuv_zeros[i : i + d, j : j + d] = np.eye(d)
                zeros_constraints.append(E_iuv_zeros)

    return ones_constraints, zeros_constraints


def _unwrap_state_vector(vec, data: FactorGraphData) -> None:
    """unwrap the state vector into the corresponding robot poses and landmark
    states"""
    raise NotImplementedError("Not sure if this works right now?")
    d = data.dimension
    num_poses = data.num_poses
    num_landmarks = data.num_landmarks

    est_poses: List[PoseVariable] = []
    est_landmarks: List[LandmarkVariable] = []

    for gt_pose in data.pose_variables:
        trans_start_idx, trans_end_idx = data.get_pose_translation_variable_indices(
            gt_pose
        )
        est_trans = vec[trans_start_idx:trans_end_idx]
        est_trans = (est_trans[0], est_trans[1])

        rot_start_idx, rot_end_idx = data.get_pose_rotation_variable_indices(gt_pose)
        est_rotation = vec[rot_start_idx:rot_end_idx].reshape(2, 2)

        I_check = est_rotation @ est_rotation.T
        assert (I_check == np.eye(2)).all(), "Is not orthogonal group"
        assert la.det(est_rotation) == 1, f"Not a rotation {est_rotation}"

        cos_theta = est_rotation[0, 0]
        sin_theta = est_rotation[1, 0]
        assert (
            est_rotation[0, 1] == -est_rotation[1, 0]
        ), f"bad rotation matrix: {est_rotation}"
        assert est_rotation[0, 0] == est_rotation[1, 1]
        est_theta = np.arctan2(sin_theta, cos_theta)

        est_pose_name = gt_pose.name + "_est"

        estimated_pose = PoseVariable(est_pose_name, est_trans, est_theta)
        est_poses.append(estimated_pose)

    for gt_landmark in data.landmark_variables:
        start_idx, end_idx = data.get_landmark_translation_variable_indices(gt_landmark)
        est_trans = vec[trans_start_idx:trans_end_idx]
        est_trans = (est_trans[0], est_trans[1])
        est_landmark_name = gt_landmark.name + "_est"
        est_landmarks.append(LandmarkVariable(est_landmark_name, est_trans))

    for pose in est_poses:
        print(pose)

    for landmark in est_landmarks:
        print(landmark)


def solve_mle_problem(data: FactorGraphData):
    """
    Takes the data describing the problem and returns the MLE solution to the
    poses and landmark positions

    args:
        data (FactorGraphData): the data describing the problem
    """
    model = gp.Model("qp")
    model.params.nonconvex = 2
    # model.params.feasibility_tol = 1e-2
    model.params.optimality_tol = 1e-9
    # model.params.time_limit = 100.0
    obj = 0

    # form objective function
    Q, D = _get_data_matrix(data)
    _check_psd(Q)
    _check_psd(D)
    full_data_matrix = la.block_diag(Q, D)

    true_vals = data.true_values_vector

    _check_psd(full_data_matrix)

    translations, rotations = add_pose_variables(model, data)
    landmarks = add_landmark_variables(model, data)
    distances = add_distance_variables(model, data, translations, landmarks)

    obj += get_distances_cost(distances, data)
    obj += get_odom_cost(translations, rotations, data)

    # add in term to move landmarks to their true locations
    for i, j in distances.keys():
        x_i = translations[i]
        l_j = landmarks[j]
        diff = x_i - l_j
        dist_ij = distances[(i, j)]
        # obj += dist_ij * dist_ij - diff.frob_norm_squared

    # pin first pose at origin
    pin_first_pose(translations[0], rotations[0])

    # perform optimization
    model.setObjective(obj, gp.GRB.MINIMIZE)
    model.optimize()

    # extract the solution
    for v in model.getVars():
        if "translation" in v.varName or "land" in v.varName:
            print("%s %g" % (v.varName, v.x))
        # print("%s %g" % (v.varName, v.x))

    # print("Obj: %g" % obj.getValue())
    # _unwrap_state_vector(state_vector.X, data)


def _get_data_matrix(data: FactorGraphData) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gets the matrix of the data for the MLE problem

    Args:
        data (FactorGraphData): the data describing the problem

    Returns:
        (np.ndarray): the pose graph data matrix (kron(M, I) from SE-Sync 18b)
        (np.ndarray): the ranging data matrix (D from my own math)
    """

    def _get_translation_laplacian(data: FactorGraphData) -> np.ndarray:
        """
        Gets the translation laplacian as defined in SE-Sync equation 13a

        args:
            data (FactorGraphData): the data describing the problem
        """
        L = np.zeros((data.num_translations, data.num_translations))
        for measure in data.odom_measurements:
            i = measure.base_pose_idx
            j = measure.to_pose_idx
            weight = measure.translation_weight

            L[i, i] += weight
            L[j, j] += weight

            assert L[i, j] == 0
            assert L[j, i] == 0
            L[i, j] -= weight
            L[j, i] -= weight

        _check_is_laplacian(L)
        return L

    def _get_connection_laplacian(data: FactorGraphData, dim: int) -> np.ndarray:
        """gets the graph connection laplacian as described in SE-Sync
        equations 14a and 14b

        args:
            data (FactorGraphData): the factor graph data
            dim (int): the dimension of the latent space (e.g. 2d or 3d)

        returns:
            np.ndarray: the connection laplacian
        """
        L = np.zeros((d * data.num_poses, d * data.num_poses))
        I_d = np.eye(d)
        for measure in data.odom_measurements:
            i = measure.base_pose_idx
            j = measure.to_pose_idx
            weight = measure.rotation_weight

            i_idx = i * dim
            j_idx = j * dim

            L[i_idx : i_idx + dim, i_idx : i_idx + dim] += weight * I_d
            L[j_idx : j_idx + dim, j_idx : j_idx + dim] += weight * I_d

            assert all(
                x == 0 for x in L[i_idx : i_idx + dim, j_idx : j_idx + dim].flatten()
            )
            assert all(
                x == 0 for x in L[j_idx : j_idx + dim, i_idx : i_idx + dim].flatten()
            )
            L[i_idx : i_idx + dim, j_idx : j_idx + dim] -= (
                weight * measure.rotation_matrix
            )
            L[j_idx : j_idx + dim, i_idx : i_idx + dim] -= (
                weight * measure.rotation_matrix.T
            )

        _check_symmetric(L)
        _check_psd(L)
        return L

    def _get_weighted_translation_matrix(data: FactorGraphData, dim: int) -> np.ndarray:
        """gets the weighted (1xd)-block-structured translation matrix as
        described in SE-Sync equation 15

        args:
            data (FactorGraphData): the factor graph data
            dim (int): the dimension of the latent space (e.g. 2d or 3d)

        returns:
            np.ndarray: the weighted translation matrix
        """
        V = np.zeros((data.num_translations, dim * data.num_poses))
        cnt = 0
        for measure in data.odom_measurements:
            cnt += 1
            i = measure.base_pose_idx
            j = measure.to_pose_idx
            assert not (i == j)
            weight = measure.translation_weight
            vect = measure.translation_vector
            assert vect.shape == (dim,) or vect.shape == (dim, 1)

            V[i, dim * i : dim * (i + 1)] += weight * vect

            assert all(x == 0 for x in V[j, dim * i : dim * (i + 1)])
            V[j, dim * i : dim * (i + 1)] -= weight * vect

        return V

    def _get_weighted_translation_sigma_matrix(
        data: FactorGraphData, dim: int
    ) -> np.ndarray:
        """gets the weighted (dxd)-block-structured translation matrix as
        described in SE-Sync equation 16

        args:
            data (FactorGraphData): the factor graph data
            dim (int): the dimension of the latent space (e.g. 2d or 3d)

        returns:
            np.ndarray: the weighted translation matrix
        """
        L = np.zeros((dim * data.num_poses, dim * data.num_poses))
        for measure in data.odom_measurements:
            i = measure.base_pose_idx
            i_idx = i * dim
            weight = measure.translation_weight

            t_vec = measure.translation_vector
            L[i_idx : i_idx + dim, i_idx : i_idx + dim] += weight * np.outer(
                t_vec, t_vec
            )

        _check_symmetric(L)
        _check_psd(L)
        return L

    def _get_weighted_range_matrix(data: FactorGraphData) -> np.ndarray:
        """gets the weighted range measurement data matrix as described in my
        notes in the latex subfolder in this repo

        args:
            data (FactorGraphData): the factor graph data

        returns:
            np.ndarray: the range measurement data matrix
        """
        mat_dim = data.num_range_measurements + 1
        D = np.zeros((mat_dim, mat_dim))
        D[0 : mat_dim - 1, 0 : mat_dim - 1] = np.diag(data.measurements_weight_vect)

        noisy_dist_vect = data.weighted_dist_measurements_vect
        D[0 : mat_dim - 1, mat_dim - 1] = -noisy_dist_vect
        D[mat_dim - 1, 0 : mat_dim - 1] = -noisy_dist_vect.T
        D[mat_dim - 1, mat_dim - 1] = data.sum_weighted_measurements_squared
        _check_symmetric(D)
        _check_psd(D)
        return D

    d = data.dimension
    num_trans = data.num_translations
    num_pose = data.num_poses

    # SE-Sync equation 13a - num_translations x num_translations
    weighted_translation_laplacian = _get_translation_laplacian(data)
    assert weighted_translation_laplacian.shape == (
        num_trans,
        num_trans,
    )

    # SE-Sync equation 14a - dn x dn
    connection_laplacian = _get_connection_laplacian(data, d)
    assert connection_laplacian.shape == (d * num_pose, d * num_pose)

    # SE-Sync equation 15 - n x dn
    V = _get_weighted_translation_matrix(data, d)
    assert V.shape == (num_trans, d * num_pose)

    # SE-Sync Equation 16
    Sigma = _get_weighted_translation_sigma_matrix(data, d)
    assert Sigma.shape == (d * num_pose, d * num_pose)

    # SE-Sync matrix M from equation 18b
    M_dim = num_pose * (d) + num_trans
    M = np.zeros((M_dim, M_dim))

    # add translation laplacian to upper left block
    M[:num_trans, :num_trans] = weighted_translation_laplacian

    # add V to upper right block
    assert all(
        x == 0 for x in M[num_trans:, num_trans : num_trans + (d * num_pose)].flatten()
    )
    M[:num_trans, num_trans:] = V

    # add V.T to lower left block
    assert all(
        x == 0 for x in M[num_trans : num_trans + (d * num_pose), :num_trans].flatten()
    )
    M[num_trans:, :num_trans] = V.T

    # add connection laplacian to bottom right block
    assert all(x == 0 for x in M[num_trans:, num_trans:].flatten())
    M[num_trans:, num_trans:] = connection_laplacian
    M[num_trans:, num_trans:] += Sigma

    # quick check on this matrix to make sure it's symmetric, PSD
    # _matprint_block(M)
    _check_psd(M)
    _check_symmetric(M)

    pose_data_matrix = np.kron(M, np.eye(d))

    # check the size of this matrix
    final_dim = data.poses_and_landmarks_dimension
    assert pose_data_matrix.shape == (
        final_dim,
        final_dim,
    ), f"Data matrix shape: {pose_data_matrix.shape}is not ({final_dim}, {final_dim})"
    _check_symmetric(pose_data_matrix)
    _check_psd(pose_data_matrix)

    range_data_matrix = _get_weighted_range_matrix(data)
    # _matprint_block(range_data_matrix)
    _check_psd(range_data_matrix)
    _check_symmetric(range_data_matrix)

    return pose_data_matrix, range_data_matrix


if __name__ == "__main__":
    file_name = "simEnvironment_grid20x20_rowCorner2_colCorner2_cellScale10_rangeProb05_rangeRadius40_falseRangeProb00_outlierProb01_loopClosureProb01_loopClosureRadius3_falseLoopClosureProb01_timestep10.fg"

    filepath = expanduser(join("~", "data", "example_factor_graphs", file_name))
    fg = parse_factor_graph_file(filepath)
    solve_mle_problem(fg)
