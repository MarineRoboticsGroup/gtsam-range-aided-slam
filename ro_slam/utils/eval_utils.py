from typing import Dict, Tuple, List
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from factor_graph.factor_graph import FactorGraphData
from factor_graph.variables import PoseVariable, LandmarkVariable

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


def plot_error(
    data: FactorGraphData,
    solved_results: Dict[str, Dict[str, np.ndarray]],
    grid_size: int,
) -> None:
    """
    Plots the error for the given data

    Args:
        data (FactorGraphData): the groundtruth data
        solved_results (Dict[str, Dict[str, np.ndarray]]): the solved values of the variables
        grid_size (int): the size of the grid

    """

    def draw_arrow(
        ax: plt.Axes,
        x: float,
        y: float,
        theta: float,
        quiver_length: float = 0.1,
        quiver_width: float = 0.01,
        color: str = "black",
    ) -> patches.FancyArrow:
        """Draws an arrow on the given axes

        Args:
            ax (plt.Axes): the axes to draw the arrow on
            x (float): the x position of the arrow
            y (float): the y position of the arrow
            theta (float): the angle of the arrow
            quiver_length (float, optional): the length of the arrow. Defaults to 0.1.
            quiver_width (float, optional): the width of the arrow. Defaults to 0.01.
            color (str, optional): color of the arrow. Defaults to "black".

        Returns:
            patches.FancyArrow: the arrow
        """
        dx = quiver_length * np.cos(theta)
        dy = quiver_length * np.sin(theta)
        return ax.arrow(
            x,
            y,
            dx,
            dy,
            head_width=quiver_length,
            head_length=quiver_length,
            width=quiver_width,
            color=color,
        )

    def draw_pose_variable(ax: plt.Axes, pose: PoseVariable):
        true_x = pose.true_x
        true_y = pose.true_y
        true_theta = pose.true_theta
        return draw_arrow(ax, true_x, true_y, true_theta, color="blue")

    def draw_pose_solution(ax: plt.Axes, translation: np.ndarray, rotation: np.ndarray):
        x = translation[0]
        y = translation[1]
        theta = float(rotation)
        return draw_arrow(ax, x, y, theta, color="red")

    def draw_landmark_variable(ax: plt.Axes, landmark: LandmarkVariable):
        true_x = landmark.true_x
        true_y = landmark.true_y
        ax.scatter(true_x, true_y, color="green", marker=(5, 2))

    def draw_landmark_solution(ax: plt.Axes, translation: np.ndarray):
        x = translation[0]
        y = translation[1]
        ax.scatter(x, y, color="red", marker=(4, 2))

    def draw_all_information(
        ax: plt.Axes,
        gt_data: FactorGraphData,
        solution_data: Dict[str, Dict[str, np.ndarray]],
        use_arrows: bool = True,
    ):
        """Draws the pose estimates and groundtruth

        Args:
            ax (plt.Axes): the axes to draw on
            gt_data (FactorGraphData): the groundtruth data
            solution_data (Dict[str, Dict[str, np.ndarray]]): the solved values of the variables
        """
        num_pose_chains = len(gt_data.pose_variables)
        pose_chain_len = len(gt_data.pose_variables[0])

        # make sure all pose chains same length
        assert all(len(x) == pose_chain_len for x in gt_data.pose_variables)
        assert num_pose_chains > 0

        for landmark in gt_data.landmark_variables:
            draw_landmark_variable(ax, landmark)
            draw_landmark_solution(ax, solution_data["landmarks"][landmark.name])

        pose_var_plot_obj: List[patches.FancyArrow] = []
        pose_sol_plot_obj: List[patches.FancyArrow] = []
        for pose_idx in range(pose_chain_len):
            for pose_chain_idx in range(num_pose_chains):
                pose = gt_data.pose_variables[pose_chain_idx][pose_idx]

                # draw inferred solution
                soln_arrow = draw_pose_solution(
                    ax,
                    solution_data["translations"][pose.name],
                    solution_data["rotations"][pose.name],
                )
                pose_sol_plot_obj.append(soln_arrow)

                # draw groundtruth solution
                var_arrow = draw_pose_variable(ax, pose)
                pose_var_plot_obj.append(var_arrow)

            plt.pause(0.1)

            if pose_idx > 10:
                # ax.remove(pose_sol_plot_obj[0])
                pose_sol_plot_obj[0].remove()
                pose_sol_plot_obj.pop(0)
                pose_var_plot_obj[0].remove()
                pose_var_plot_obj.pop(0)

        plt.close()

    if data.num_poses < 50:
        return

    # set up plot
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.grid(True)
    plt.xticks(range(grid_size + 1))
    plt.yticks(range(grid_size + 1))
    ax.set_xlim(-0.5, grid_size + 0.5)
    ax.set_ylim(-0.5, grid_size + 0.5)

    # draw all poses to view static image result
    draw_all_information(ax, data, solved_results)

    # TODO show animation of solution
