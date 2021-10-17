from typing import Dict, Tuple, List
import numpy as np

from ro_slam.utils.matrix_utils import (
    get_theta_from_matrix,
    _check_rotation_matrix,
    get_theta_from_matrix_so_projection,
)


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
    with open(filepath, "w") as f:
        translations = solved_results["translations"]
        rotations = solved_results["rotations"]
        for pose_key in translations.keys():
            trans_solve = translations[pose_key]
            theta_solve = rotations[pose_key]
            assert theta_solve.size == 1

            trans_string = np.array2string(trans_solve, precision=1, floatmode="fixed")
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
            status = f"State {landmark_key}" + f" | Translation: {landmark_string}\n"
            f.write(status)

        f.write(f"Is optimization successful? {result.is_success()}\n")
        f.write(f"optimal cost: {result.get_optimal_cost()}")

    print(f"Results saved to: {filepath}\n")


def get_solved_values(
    result,
    translations: Dict[str, np.ndarray],
    rotations: Dict[str, np.ndarray],
    landmarks: Dict[str, np.ndarray],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
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

    return (
        solved_translations,
        solved_rotations,
        solved_landmarks,
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
