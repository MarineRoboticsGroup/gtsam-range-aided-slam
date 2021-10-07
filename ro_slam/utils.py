import numpy as np
import scipy.linalg as la  # type: ignore
from typing import List, Tuple


def get_theta_from_matrix(mat: np.ndarray):
    """
    Returns theta from a matrix M
    """
    _check_square(mat)
    return np.arctan2(mat[1, 0], mat[0, 0])


def _print_eigvals(
    M: np.ndarray, name: str = None, print_eigvec: bool = False, symmetric: bool = True
):
    """print the eigenvalues of a matrix"""

    if name is not None:
        print(name)

    if print_eigvec:
        # get the eigenvalues of the matrix
        if symmetric:
            eigvals, eigvecs = la.eigh(M)
        else:
            eigvals, eigvecs = la.eig(M)

        # sort the eigenvalues and eigenvectors
        idx = eigvals.argsort()[::1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        print(f"eigenvectors: {eigvecs}")
    else:
        if symmetric:
            eigvals = la.eigvalsh(M)
        else:
            eigvals = la.eigvals(M)
        print(f"eigenvalues\n{eigvals}")

    print("\n\n\n")


def _check_rotation_matrix(R: np.ndarray):
    """
    Checks that R is a rotation matrix.

    Args:
        R (np.ndarray): the candidate rotation matrix
    """
    d = R.shape[0]
    is_orthogonal = np.allclose(R @ R.T, np.eye(d), rtol=1e-3, atol=1e-3)
    if not is_orthogonal:
        print(f"R not orthogonal: {R @ R.T}")


def _check_square(mat: np.ndarray):
    assert mat.shape[0] == mat.shape[1], "matrix must be square"


def _check_symmetric(mat):
    assert np.allclose(mat, mat.T)


def _check_psd(mat: np.ndarray):
    """Checks that a matrix is positive semi-definite"""
    assert isinstance(mat, np.ndarray)
    assert (
        np.min(la.eigvals(mat)) + 1e-1 >= 0.0
    ), f"min eigenvalue is {np.min(la.eigvals(mat))}"


def _check_is_laplacian(L: np.ndarray):
    """Checks that a matrix is a Laplacian based on well-known properties

    Must be:
        - symmetric
        - ones vector in null space of L
        - no negative eigenvalues

    Args:
        L (np.ndarray): the candidate Laplacian
    """
    assert isinstance(L, np.ndarray)
    _check_symmetric(L)
    _check_psd(L)
    ones = np.ones(L.shape[0])
    zeros = np.zeros(L.shape[0])
    assert np.allclose(L @ ones, zeros), f"L @ ones != zeros: {L @ ones}"


def _matprint_block(mat, fmt="g"):
    col_maxes = [max([len(("{:" + fmt + "}").format(x)) for x in col]) for col in mat.T]
    num_col = mat.shape[1]
    row_spacer = ""
    for _ in range(num_col):
        row_spacer += "__ __ __ "
    for j, x in enumerate(mat):
        if j % 2 == 0:
            print(row_spacer)
            print("")
        for i, y in enumerate(x):
            if i % 2 == 1:
                print(("{:" + str(col_maxes[i]) + fmt + "}").format(y), end=" | ")
            else:
                print(("{:" + str(col_maxes[i]) + fmt + "}").format(y), end="  ")
        print("")

    print(row_spacer)
    print("\n\n\n")


def _general_kron(a, b):
    """
    Returns a CVXPY Expression representing the Kronecker product of a and b.

    At most one of "a" and "b" may be CVXPY Variable objects.

    :param a: 2D numpy ndarray, or a CVXPY Variable with a.ndim == 2
    :param b: 2D numpy ndarray, or a CVXPY Variable with b.ndim == 2
    """
    expr = np.kron(a, b)
    num_rows = expr.shape[0]
    rows = [cp.hstack(expr[i, :]) for i in range(num_rows)]
    full_expr = cp.vstack(rows)
    return full_expr
