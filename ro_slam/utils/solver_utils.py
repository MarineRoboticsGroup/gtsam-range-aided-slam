from typing import Union, Dict, Optional
import pickle
from os.path import isfile
import numpy as np
import attr


from pydrake.solvers.mathematicalprogram import MathematicalProgram  # type: ignore
from pydrake.solvers.ipopt import IpoptSolver
from pydrake.solvers.snopt import SnoptSolver
from pydrake.solvers.gurobi import GurobiSolver
from pydrake.solvers.mosek import MosekSolver

from ro_slam.utils.matrix_utils import (
    get_theta_from_matrix,
    _check_rotation_matrix,
    get_theta_from_matrix_so_projection,
)


@attr.s(frozen=True)
class SolverParams:
    solver: str = attr.ib()
    verbose: bool = attr.ib()
    save_results: bool = attr.ib()
    use_socp_relax: bool = attr.ib()
    use_orthogonal_constraint: bool = attr.ib()
    init_technique: str = attr.ib()
    custom_init_file: Optional[str] = attr.ib(default=None)


@attr.s(frozen=True)
class VariableValues:
    translations: Dict[str, np.ndarray] = attr.ib()
    rotations: Dict[str, np.ndarray] = attr.ib()
    landmarks: Dict[str, np.ndarray] = attr.ib()
    distances: Dict[str, np.ndarray] = attr.ib()


@attr.s(frozen=True)
class SolverResults:
    variables: VariableValues = attr.ib()
    total_time: float = attr.ib()
    solved: bool = attr.ib()


def print_state(
    result,
    translations: Dict[str, np.ndarray],
    rotations: Dict[str, np.ndarray],
    pose_key: str,
):
    """
    Prints the current state of the result

    Args:
        result (MathematicalProgram): the result of the solution
        translations (List[np.ndarray]): the translations
        rotations (List[np.ndarray]): the rotations
        pose_key (str): the key of the pose to print
    """
    trans_solve = result.GetSolution(translations[pose_key]).round(decimals=2)
    rot_solve = result.GetSolution(rotations[pose_key])
    theta_solve = get_theta_from_matrix(rot_solve)

    trans_string = np.array2string(trans_solve, precision=1, floatmode="fixed")

    status = (
        f"State {pose_key}"
        + f" | Translation: {trans_string}"
        + f" | Rotation: {theta_solve:.2f}"
    )
    print(status)


def save_results_to_file(
    result,
    solved_results: Dict[str, Dict[str, np.ndarray]],
    filepath: str,
):
    """
    Saves the results to a file

    Args:
        result (Drake Results): the result of the solution
        solved_results (Dict[str, Dict[str, np.ndarray]]): the solved values of the variables
        filepath (str): the path to save the results to
    """
    allowed_extensions = [".pickle", ".txt"]

    if filepath.endswith(".pickle"):
        pickle_file = open(filepath, "wb")
        pickle.dump(solved_results, pickle_file)
        solve_info = {
            "success": result.is_success(),
            "optimal_cost": result.get_optimal_cost(),
        }
        pickle.dump(solve_info, pickle_file)
        pickle_file.close()

    elif filepath.endswith(".txt"):
        with open(filepath, "w") as f:
            translations = solved_results["translations"]
            rotations = solved_results["rotations"]
            for pose_key in translations.keys():
                trans_solve = translations[pose_key]
                theta_solve = rotations[pose_key]
                assert theta_solve.size == 1

                trans_string = np.array2string(
                    trans_solve, precision=1, floatmode="fixed"
                )
                status = (
                    f"State {pose_key}"
                    + f" | Translation: {trans_string}"
                    + f" | Rotation: {theta_solve:.2f}\n"
                )
                f.write(status)

            landmarks = solved_results["landmarks"]
            for landmark_key in landmarks.keys():
                landmark_solve = landmarks[landmark_key]

                landmark_string = np.array2string(
                    landmark_solve, precision=1, floatmode="fixed"
                )
                status = (
                    f"State {landmark_key}" + f" | Translation: {landmark_string}\n"
                )
                f.write(status)

            f.write(f"Is optimization successful? {result.is_success()}\n")
            f.write(f"optimal cost: {result.get_optimal_cost()}")

    else:
        raise ValueError(
            f"The file extension {filepath.split('.')[-1]} is not supported. "
        )

    print(f"Results saved to: {filepath}\n")


def get_solved_values(
    result,
    translations: Dict[str, np.ndarray],
    rotations: Dict[str, np.ndarray],
    landmarks: Dict[str, np.ndarray],
    distances: Dict[str, np.ndarray],
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
    solved_translations = {
        key: result.GetSolution(translations[key]) for key in translations.keys()
    }
    solved_rotations = {
        key: np.asarray(
            get_theta_from_matrix_so_projection(result.GetSolution(rotations[key]))
        )
        for key in rotations.keys()
    }
    solved_landmarks = {
        key: result.GetSolution(landmarks[key]) for key in landmarks.keys()
    }
    solved_distances = {
        key: result.GetSolution(distances[key]) for key in distances.keys()
    }

    return SolverResults(
        VariableValues(
            translations=solved_translations,
            rotations=solved_rotations,
            landmarks=solved_landmarks,
            distances=solved_distances,
        ),
        total_time=result.get_total_time(),
        solved=result.is_success(),
    )


def check_rotations(result, rotations: Dict[str, np.ndarray]):
    """
    checks that results are valid rotations

    Args:
        result (Drake Result Object): the result of the solution
        rotations (Dict[str, np.ndarray]): the rotation variables
    """
    for rot_key in rotations.keys():
        rot_result = result.GetSolution(rotations[rot_key])
        _check_rotation_matrix(rot_result)


def get_solver(
    solver_name: str,
) -> Union[IpoptSolver, SnoptSolver, GurobiSolver, MosekSolver]:
    """
    Returns the solver for the given name
    """
    if solver_name == "ipopt":
        return IpoptSolver()
    elif solver_name == "snopt":
        return SnoptSolver()
    elif solver_name == "gurobi":
        return GurobiSolver()
    elif solver_name == "mosek":
        return MosekSolver()
    else:
        raise ValueError(f"Unknown solver: {solver_name}")


def set_solver_verbose(
    model: MathematicalProgram,
    solver: Union[IpoptSolver, SnoptSolver, GurobiSolver, MosekSolver],
):
    """Sets the given solver to verbose output

    Args:
        solver (Union[IpoptSolver, SnoptSolver, GurobiSolver, MosekSolver]): [description]
    """
    if isinstance(solver, IpoptSolver):
        raise NotImplementedError("IpoptSolver verbose logging not implemented")
    elif isinstance(solver, SnoptSolver):
        print("SnoptSolver verbose logging not implemented")
    elif isinstance(solver, GurobiSolver):
        model.SetSolverOption(solver.solver_id(), "OutputFlag", True)
        model.SetSolverOption(solver.solver_id(), "LogToConsole", True)
    elif isinstance(solver, MosekSolver):
        solver.set_stream_logging(True, "/home/alan/mosek_log.out")
    else:
        raise ValueError("Unknown solver")


def load_custom_init_file(file_path: str) -> SolverResults:
    """Loads the custom init file

    Args:
        file_path (str): [description]
    """

    assert isfile(file_path), f"File {file_path} does not exist"
    assert file_path.endswith(".pickle"), f"File {file_path} must end with '.pickle'"

    with open(file_path, "rb") as f:
        init_dict = pickle.load(f)
        assert isinstance(init_dict, SolverResults), "Loaded object is not a dict"
        return init_dict
