import cvxpy as cp
import numpy as np
from os.path import expanduser, join
from typing import List, Tuple
import matplotlib.pyplot as plt
import scipy.linalg as la

from factor_graph.parse_factor_graph import parse_factor_graph_file
from factor_graph.factor_graph import (
    OdomMeasurement,
    RangeMeasurement,
    PoseVariable,
    LandmarkVariable,
    PosePrior,
    LandmarkPrior,
    FactorGraphData,
)
from utils import (
    _block_diag,
    _print_eigvals,
    _check_is_laplacian,
    _check_symmetric,
    _check_psd,
    _general_kron,
    _matprint_block,
)


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
        for measure in data.odometry_measurements:
            i = measure.base_pose_index
            j = measure.to_pose_index
            weight = measure.translation_weight

            L[i, i] += weight
            L[j, j] += weight

            assert L[i, j] == 0
            assert L[j, i] == 0
            L[i, j] -= weight
            L[j, i] -= weight

        _check_is_laplacian(L)
        return L

    def _get_rotation_laplacian(data: FactorGraphData) -> np.ndarray:
        """Get the rotation laplacian as defined in SE-Sync equation 13b

        Args:
            data (FactorGraphData): the problem data

        Returns:
            np.ndarray: the rotation laplacian
        """
        L = np.zeros((data.num_poses, data.num_poses))
        for measurement in data.odom_measurements:
            i = measurement.base_pose
            j = measurement.to_pose
            weight = measurement.rotation_weight

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
        for measure in data.odometry_measurements:
            i = measure.base_pose_index
            j = measure.to_pose_index
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
        for measure in data.odometry_measurements:
            cnt += 1
            i = measure.base_pose_index
            j = measure.to_pose_index
            assert not (i == j)
            weight = measure.translation_weight
            vect = measure.translation_vector
            assert vect.shape == (dim,)

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
        for measure in data.odometry_measurements:
            i = measure.base_pose_index
            i_idx = i * dim
            weight = measure.translation_weight

            t_vec = measure.translation_vector
            L[i_idx : i_idx + dim, i_idx : i_idx + dim] += weight * np.outer(
                t_vec, t_vec
            )

        _check_symmetric(L)
        _check_psd(L)
        return L

    # TODO incorporate weights into this - right now uniform weighting!
    def _get_weighted_range_matrix(data: FactorGraphData) -> np.ndarray:
        """gets the weighted range measurement data matrix as described in my
        notes in the latex subfolder in this repo

        args:
            data (FactorGraphData): the factor graph data

        returns:
            np.ndarray: the range measurement data matrix
        """
        mat_dim = data.num_range_measurements + 1
        D = np.eye(mat_dim)

        noisy_dist_vect = data.dist_measurements_vect
        D[0 : mat_dim - 1, mat_dim - 1] = -noisy_dist_vect
        D[mat_dim - 1, 0 : mat_dim - 1] = -noisy_dist_vect.T
        D[mat_dim - 1, mat_dim - 1] = np.dot(noisy_dist_vect, noisy_dist_vect)
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
    M[:num_trans, num_trans : num_trans + (d * num_pose)] = V

    # add V.T to lower left block
    assert all(
        x == 0 for x in M[num_trans : num_trans + (d * num_pose), :num_trans].flatten()
    )
    M[num_trans : num_trans + (d * num_pose), :num_trans] = V.T

    # add connection laplacian to bottom right block
    assert all(x == 0 for x in M[num_trans:, num_trans:].flatten())
    M[num_trans:, num_trans:] = connection_laplacian
    M[num_trans:, num_trans:] += Sigma

    # quick check on this matrix to make sure it's symmetric, PSD
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
    _check_psd(range_data_matrix)
    _check_symmetric(range_data_matrix)

    return pose_data_matrix, range_data_matrix


def _plot_constraint_matrices(Lambda, K, Q, data):
    """
    Plots the sparsity pattern of these matrices associated with the Lagrangian
    dual problem
    """
    range_ind_start = data.num_poses * data.dimension - 0.5
    range_ind_end = data.num_translations * data.dimension - 0.5
    rot_ind_start = range_ind_end
    rot_ind_end = range_ind_end + data.num_poses * data.dimension ** 2

    # _matprint_block(Lambda.value)
    plt.spy(Lambda.value)
    plt.title("Lambda Sparsity")

    # lines around range constraints
    plt.hlines([range_ind_start, range_ind_end], 0, 90)
    plt.vlines([range_ind_start, range_ind_end], 0, 90)

    # lines around rotation constraints
    plt.hlines([rot_ind_start, rot_ind_end], 0, 90)
    plt.vlines([rot_ind_start, rot_ind_end], 0, 90)

    plt.show()

    plt.spy(K.value)
    plt.title("K Sparsity")

    # lines around range constraints
    plt.hlines([range_ind_start, range_ind_end], 0, 90)
    plt.vlines([range_ind_start, range_ind_end], 0, 90)

    # lines around rotation constraints
    plt.hlines([rot_ind_start, rot_ind_end], 0, 90)
    plt.vlines([rot_ind_start, rot_ind_end], 0, 90)
    plt.show()

    plt.spy(Lambda.value + K.value)
    plt.title("Lambda and K Sparsity")
    # lines around range constraints
    plt.hlines([range_ind_start, range_ind_end], 0, 90)
    plt.vlines([range_ind_start, range_ind_end], 0, 90)

    # lines around rotation constraints
    plt.hlines([rot_ind_start, rot_ind_end], 0, 90)
    plt.vlines([rot_ind_start, rot_ind_end], 0, 90)
    plt.show()


def _get_range_constraint_matrix(
    data: FactorGraphData,
) -> Tuple[cp.Variable, List[cp.Variable]]:
    """gets the range constraint matrix for the lagrangian dual

    args:
        data (FactorGraphData): the data

    returns:
        cp.Variable: the range constraint matrix
        List[cp.Variable]: the list of the Lagrange multipliers
    """
    d = data.dimension
    mat_dim = data.poses_and_landmarks_dimension
    K = np.zeros((mat_dim, mat_dim))

    # we're going to keep these around just in case I end up needing them
    lagrange_multipliers = []

    # iterate over all range measurements
    for measure in data.range_measurements:

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
        assert isinstance(j_start, int) and isinstance(j_end, int)
        assert j_start < j_end
        assert j_start >= data.num_poses * d
        assert (
            j_end <= data.num_translations * d
        ), f"{j_end} > {data.num_translations * d}"

        # this matrix represents the quadratic constraint between the
        # distance variable d_ij and the two translations it is relating
        # (t_i, t_j) -> d_ij^2 = ||t_i - t_j||^2
        Kij = np.zeros(K.shape)

        # The block matrices on the diagonals corresponding to the
        # translations are identity matrices and the ones on the
        # off-diagonals are negative identity matrices
        Kij[i_start:i_end, i_start:i_end] = np.eye(d)
        Kij[j_start:j_end, j_start:j_end] = np.eye(d)

        Kij[i_start:i_end, j_start:j_end] = -np.eye(d)
        Kij[j_start:j_end, i_start:i_end] = -np.eye(d)

        # sanity check this
        _check_symmetric(Kij)

        # Finally, we scale this by this constraints corresponding lagrange
        # multiplier and add it all together to the big matrix where we are
        # summing these constraints
        lagrange_multipliers.append(cp.Variable())
        K += Kij * lagrange_multipliers[-1]

    return K, lagrange_multipliers


def _get_rotations_constraint_matrix(data: FactorGraphData) -> cp.Variable:
    """gets the rotation constraint matrix for the lagrangian dual
    enforcing that R.T @ R = I for all rotations

    args:
        data (FactorGraphData): the data

    returns:
        cp.Variable: the rotation constraint matrix
    """

    d = data.dimension
    I_d = np.eye(d)
    lambdas = []

    # add in an offset to ignore all of the translation constraints
    num_trans = data.num_translations
    translation_offset = np.zeros((d * num_trans, d * num_trans))
    lambdas.append(translation_offset)

    # add lagrange multipliers
    for _ in range(data.num_poses):
        lam = cp.Variable((d, d), symmetric=True)

        # add in kronecker product because we're vectorizing the rotations
        lam = _general_kron(lam, I_d)
        lambdas.append(lam)

    Lambda = _block_diag(lambdas)
    return Lambda


def solve_lagrangian_dual(data: FactorGraphData):
    """
    Solves the lagrangian dual of the MLE problem

    args:
        data (FactorGraphData): the data describing the problem
    """
    K, range_lagrange_multipliers = _get_range_constraint_matrix(data)
    Lambda = _get_rotations_constraint_matrix(data)
    Q, D = _get_data_matrix(data)
    _check_psd(Q)
    _check_psd(D)

    # Nu is the lagrange multipliers arranged along a diagonal plus a zero for
    # padding at the end
    Nu = cp.diag(cp.hstack(range_lagrange_multipliers + [0]))
    nu_len = data.num_range_measurements + 1
    assert Nu.shape == (nu_len, nu_len)

    assert K.shape == Q.shape, f"K.shape: {K.shape}, Q.shape: {Q.shape}"
    assert Lambda.shape == Q.shape, f"Lambda.shape: {Lambda.shape}, Q.shape: {Q.shape}"

    cost = cp.trace(Lambda)
    obj = cp.Maximize(cost)
    M1 = Q - Lambda - K
    M2 = D + Nu
    cons = []
    cons.append(M1 >> 0)
    cons.append(M2 >> 0)
    problem = cp.Problem(obj, cons)

    print("\n\n")
    print("------------------------")
    print("Solving the dual problem")
    # problem.solve(verbose=True, solver=cp.CVXOPT, use_indirect=True)
    problem.solve(verbose=True, solver=cp.MOSEK)

    return M1.value, M2.value


if __name__ == "__main__":
    file_name = "simEnvironment_grid20x20_rowCorner2_colCorner2_cellScale10_rangeProb05_rangeRadius10_falseRangeProb01_outlierProb01_loopClosureProb01_loopClosureRadius3_falseLoopClosureProb01_timestep10.fg"
    # file_name = "imEnvironment_grid20x20_rowCorner2_colCorner2_cellScale10_rangeProb05_rangeRadius40_falseRangeProb01_outlierProb01_loopClosureProb01_loopClosureRadius3_falseLoopClosureProb01_timestep100.txt"
    filepath = expanduser(join("~", "data", "example_factor_graphs", file_name))
    fg = parse_factor_graph_file(filepath)

    print(fg)

    M1, M2 = solve_lagrangian_dual(fg)
    _print_eigvals(M1, name="M1")
    _print_eigvals(M2, name="M2")

    _check_symmetric(M1)
    _check_psd(M1)
    _check_symmetric(M2)
    _check_psd(M2)

    Q, D = _get_data_matrix(fg)
    # _print_eigvals(Q, name="Q", print_eigvec=True)
    _print_eigvals(D, name="D", print_eigvec=True)
    # _matprint_block(D)
    # plt.spy(Q)
    # plt.show()
    # plt.spy(D)
    # plt.show()

    full_data_matrix = la.block_diag(Q, D)

    lmm = la.block_diag(M1, M2)
    print("here")
    print(la.eigvals(lmm))
    plt.spy(lmm)
    plt.show()
    left_part = lmm[:, :-1]
    right_part = lmm[:, -1]
    xstar = la.pinv(left_part) @ right_part
    print(xstar)
    # _check_symmetric(full_data_matrix)
    # _print_eigvals(full_data_matrix)
    plt.spy(full_data_matrix)
    # pts = [21.5, 31.5]
    # plt.hlines(pts, 0, 90)
    # plt.vlines(pts, 0, 90)
    plt.show()
