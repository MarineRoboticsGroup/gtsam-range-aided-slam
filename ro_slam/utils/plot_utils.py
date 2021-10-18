from typing import Dict, Tuple, List, Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from factor_graph.factor_graph import FactorGraphData
from factor_graph.variables import PoseVariable, LandmarkVariable
from ro_slam.utils.circle_utils import Arc, Circle, CircleIntersection, Point

colors = ["red", "green", "blue", "orange", "purple", "black", "cyan"]


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
        num_landmarks = len(gt_data.landmark_variables)

        range_measures = gt_data.range_measurements
        range_measure_dict = {
            measure.association: measure.dist for measure in range_measures
        }

        # make sure all pose chains same length
        assert all(len(x) == pose_chain_len for x in gt_data.pose_variables)
        assert num_pose_chains > 0

        for landmark in gt_data.landmark_variables:
            draw_landmark_variable(ax, landmark)
            draw_landmark_solution(ax, solution_data["landmarks"][landmark.name])

        pose_var_plot_obj: List[patches.FancyArrow] = []
        pose_sol_plot_obj: List[patches.FancyArrow] = []
        dist_arcs_plot_obj: List[List[patches.Arc]] = [[] for _ in range(num_landmarks)]
        range_circles: List[CircleIntersection] = [
            CircleIntersection() for _ in range(num_landmarks)
        ]
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

                # draw arc to inferred landmarks
                for landmark_idx, landmark in enumerate(gt_data.landmark_variables):
                    soln_pose_center = solution_data["translations"][pose.name]
                    soln_landmark_center = solution_data["landmarks"][landmark.name]
                    range_key = (pose.name, landmark.name)
                    if range_key in range_measure_dict:
                        arc_radius = range_measure_dict[range_key]
                        dist_circle = Circle(
                            Point(soln_pose_center[0], soln_pose_center[1]), arc_radius
                        )
                        range_circles[landmark_idx].add_circle(dist_circle)
                    range_circles[landmark_idx].draw_intersection(
                        ax, color=colors[landmark_idx]
                    )
                    # range_circles[landmark_idx].draw_circles(
                    #     ax, color=colors[landmark_idx]
                    # )

                # draw groundtruth solution
                var_arrow = draw_pose_variable(ax, pose)
                pose_var_plot_obj.append(var_arrow)

            plt.pause(0.1)
            ax.patches = []

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


def draw_arc_patch(
    arc: Arc,
    ax: plt.Axes,
    resolution: int = 50,
    color: str = "black",
) -> patches.Polygon:
    """Draws an arc as a generic patches.Polygon

    Args:
        arc (Arc): the arc to draw
        ax (plt.Axes): the axes to draw the arc on
        resolution (int, optional): the resolution of the arc. Defaults to
        50.
        color (str, optional): the color of the arc. Defaults to "black".

    Returns:
        patches.Polygon: the arc
    """
    center = arc.center
    radius = arc.radius
    theta1, theta2 = arc.thetas
    # generate the points
    theta = np.linspace((theta1), (theta2), resolution)
    points = np.vstack(
        (radius * np.cos(theta) + center[0], radius * np.sin(theta) + center[1])
    )
    # build the polygon and add it to the axes
    poly = patches.Polygon(points.T, closed=True)
    ax.add_patch(poly)
    return poly
