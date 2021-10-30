from typing import Union, Dict, Optional, Tuple
import pickle
from os.path import isfile
import numpy as np
import attr


from ro_slam.utils.matrix_utils import (
    get_theta_from_rotation_matrix,
    get_theta_from_transformation_matrix,
    get_translation_from_transformation_matrix,
    _check_rotation_matrix,
    _check_transformation_matrix,
)


@attr.s(frozen=True)
class QcqpSolverParams:
    solver: str = attr.ib()
    verbose: bool = attr.ib()
    save_results: bool = attr.ib()
    use_socp_relax: bool = attr.ib()
    use_orthogonal_constraint: bool = attr.ib()
    init_technique: str = attr.ib()
    custom_init_file: Optional[str] = attr.ib(default=None)


@attr.s(frozen=True)
class GtsamSolverParams:
    verbose: bool = attr.ib()
    save_results: bool = attr.ib()
    init_technique: str = attr.ib()
    custom_init_file: Optional[str] = attr.ib()


@attr.s(frozen=True)
class VariableValues:
    poses: Dict[str, np.ndarray] = attr.ib()
    # translations: Dict[str, np.ndarray] = attr.ib()
    # rotations: Dict[str, np.ndarray] = attr.ib()
    landmarks: Dict[str, np.ndarray] = attr.ib()
    distances: Optional[Dict[Tuple[str, str], np.ndarray]] = attr.ib()

    @poses.validator
    def _check_poses(self, attribute, value: Dict[str, np.ndarray]):
        for pose in value.values():
            _check_transformation_matrix(pose)

    @property
    def rotations(self):
        return {
            key: get_theta_from_transformation_matrix(value)
            for key, value in self.poses.items()
        }

    @property
    def translations(self):
        return {
            key: get_translation_from_transformation_matrix(value)
            for key, value in self.poses.items()
        }


@attr.s(frozen=True)
class SolverResults:
    variables: VariableValues = attr.ib()
    total_time: float = attr.ib()
    solved: bool = attr.ib()

    @property
    def translations(self):
        return self.variables.translations

    @property
    def rotations(self):
        return self.variables.rotations

    @property
    def landmarks(self):
        return self.variables.landmarks

    @property
    def distances(self):
        return self.variables.distances


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
    theta_solve = get_theta_from_rotation_matrix(rot_solve)

    trans_string = np.array2string(trans_solve, precision=1, floatmode="fixed")

    status = (
        f"State {pose_key}"
        + f" | Translation: {trans_string}"
        + f" | Rotation: {theta_solve:.2f}"
    )
    print(status)


def save_results_to_file(
    result,
    solved_results: SolverResults,
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
            translations = solved_results.translations
            rotations = solved_results.rotations
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

            landmarks = solved_results.landmarks
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


def load_custom_init_file(file_path: str) -> VariableValues:
    """Loads the custom init file

    Args:
        file_path (str): [description]
    """

    assert isfile(file_path), f"File {file_path} does not exist"
    assert file_path.endswith(".pickle"), f"File {file_path} must end with '.pickle'"

    print(f"Loading custom init file: {file_path}")
    with open(file_path, "rb") as f:
        init_dict = pickle.load(f)
        if isinstance(init_dict, SolverResults):
            return init_dict.variables
        elif isinstance(init_dict, VariableValues):
            return init_dict
        else:
            raise ValueError(f"Unknown type: {type(init_dict)}")