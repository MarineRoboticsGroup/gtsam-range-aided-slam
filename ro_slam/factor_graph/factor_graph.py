from typing import Tuple, List
import numpy as np
import attr


@attr.s(frozen=True)
class PoseVariable:
    """A variable which is a robot pose

    Arguments:
        name (str): the name of the variable (defines the frame)
        true_position (Tuple[float, float]): the true position of the robot
        true_theta (float): the true orientation of the robot
    """

    name: str = attr.ib()
    true_position: Tuple[float, float] = attr.ib()
    true_theta: float = attr.ib()

    @property
    def rotation_matrix(self):
        """
        Get the rotation matrix for the measurement
        """
        return np.array(
            [
                [np.cos(self.true_theta), -np.sin(self.true_theta)],
                [np.sin(self.true_theta), np.cos(self.true_theta)],
            ]
        )


@attr.s(frozen=True)
class LandmarkVariable:
    """A variable which is a landmark

    Arguments:
        name (str): the name of the variable
        true_position (Tuple[float, float]): the true position of the landmark
    """

    name: str = attr.ib()
    true_position: Tuple[float, float] = attr.ib()


@attr.s(frozen=True)
class OdomMeasurement:
    """
    An odometry measurement

    base_pose (str): the name of the base pose which the measurement is in the
        reference frame of
    local_pose (str): the name of the pose the measurement is to
    position (Tuple[float,float]): the change in x and y
    theta (float): the change in theta
    covariance (np.ndarray): a 3x3 covariance matrix
    """

    base_pose: str = attr.ib()
    to_pose: str = attr.ib()
    x: float = attr.ib()
    y: float = attr.ib()
    theta: float = attr.ib()
    translation_weight: float = attr.ib()
    rotation_weight: float = attr.ib()

    @property
    def rotation_matrix(self):
        """
        Get the rotation matrix for the measurement
        """
        return np.array(
            [
                [np.cos(self.theta), -np.sin(self.theta)],
                [np.sin(self.theta), np.cos(self.theta)],
            ]
        )

    @property
    def translation_vector(self):
        """
        Get the translation vector for the measurement
        """
        return np.array([self.x, self.y])

    @property
    def base_pose_index(self) -> int:
        """
        Get the index of the base pose
        """
        return int(self.base_pose[1:])

    @property
    def to_pose_index(self) -> int:
        """
        Get the index of the to pose
        """
        return int(self.to_pose[1:])


@attr.s(frozen=True)
class RangeMeasurement:
    """A range measurement

    Arguments:
        var1 (str): one variable the measurement is associated with
        var2 (str): the other variable the measurement is associated with
        dist (float): The measured range
        stddev (float): The standard deviation
    """

    var1: str = attr.ib()
    var2: str = attr.ib()
    dist: float = attr.ib()
    stddev: float = attr.ib()

    @property
    def weight(self):
        """
        Get the weight of the measurement
        """
        return 1 / (self.stddev ** 2)


@attr.s(frozen=True)
class PosePrior:
    """A prior on the robot pose

    Arguments:
        name (str): the name of the pose variable
        position (Tuple[float, float]): the prior of the position
        theta (float): the prior of the theta
        covariance (np.ndarray): the covariance of the prior
    """

    _name: str = attr.ib()
    _position: Tuple[float, float] = attr.ib()
    _theta: float = attr.ib()
    _covariance: np.ndarray = attr.ib()


@attr.s(frozen=True)
class LandmarkPrior:
    """A prior on the landmark

    Arguments:
        name (str): the name of the landmark variable
        position (Tuple[float, float]): the prior of the position
        covariance (np.ndarray): the covariance of the prior
    """

    _name: str = attr.ib()
    _position: Tuple[float, float] = attr.ib()
    _covariance: np.ndarray = attr.ib()


@attr.s
class FactorGraphData:
    """
    Just a container for the data in a FactorGraph. Only considers standard
    gaussian measurements.

    Args:
        pose_variables (List[PoseVariable]): a list of the pose variables
        landmark_variables (List[LandmarkVariable]): a list of the landmarks
        odometry_measurements (List[OdomMeasurement]): a list of odometry
            measurements
        range_measurements (List[RangeMeasurement]): a list of range
            measurements
        pose_priors (List[PosePrior]): a list of the pose priors
        landmark_priors (List[LandmarkPrior]): a list of the landmark priors
    """

    pose_variables: List[PoseVariable] = attr.ib()
    landmark_variables: List[LandmarkVariable] = attr.ib()
    odometry_measurements: List[OdomMeasurement] = attr.ib()
    range_measurements: List[RangeMeasurement] = attr.ib()
    pose_priors: List[PosePrior] = attr.ib()
    landmark_priors: List[LandmarkPrior] = attr.ib()
    _dimension: int = attr.ib()

    def __str__(self):
        line = "Factor Graph Data\n"

        # add pose variables
        line += f"Pose Variables: {len(self.pose_variables)}\n"
        for x in self.pose_variables:
            line += f"{x}\n"
        line += "\n"

        # add landmarks
        line += f"Landmark Variables: {len(self.landmark_variables)}\n"
        for x in self.landmark_variables:
            line += f"{x}\n"
        line += "\n"

        # add odometry measurements
        line += f"Odometry Measurements: {len(self.odometry_measurements)}\n"
        for x in self.odometry_measurements:
            line += f"{x}\n"
        line += "\n"

        # add range measurements
        line += f"Range Measurements: {len(self.range_measurements)}\n"
        for x in self.range_measurements:
            line += f"{x}\n"
        line += "\n"

        # add pose priors
        line += f"Pose Priors: {len(self.pose_priors)}\n"
        for x in self.pose_priors:
            line += f"{x}\n"
        line += "\n"

        # add landmark priors
        line += f"Landmark Priors: {len(self.landmark_priors)}\n"
        for x in self.landmark_priors:
            line += f"{x}\n"
        line += "\n"

        # add dimension
        line += f"Dimension: {self._dimension}\n\n"
        return line

    @property
    def num_poses(self):
        return len(self.pose_variables)

    @property
    def true_values_vector(self) -> np.ndarray:
        """
        returns the true values in a vectorized form
        """
        vect: List[float] = []

        # add in pose translations
        for pose in self.pose_variables:
            x, y = pose.true_position
            vect.append(x)
            vect.append(y)

        # add in landmark positions
        for pos in self.landmark_variables:
            x, y = pos.true_position
            vect.append(x)
            vect.append(y)

        # add in rotation measurements
        for pose in self.pose_variables:
            rot = pose.rotation_matrix.T.flatten()
            for v in rot:
                vect.append(v)

        # add in distance variables
        for range_measurement in self.range_measurements:
            vect.append(range_measurement.dist)

        vect.append(1.0)
        return np.array(vect)

    @property
    def num_translations(self):
        return self.num_poses + self.num_landmarks

    @property
    def num_landmarks(self):
        return len(self.landmark_variables)

    @property
    def dimension(self):
        return self._dimension

    @property
    def poses_and_landmarks_dimension(self):
        d = self.dimension

        # size due to translations
        n_trans = self.num_translations
        mat_dim = n_trans * d

        # size due to rotations
        n_pose = self.num_poses
        mat_dim += n_pose * d * d

        return mat_dim

    @property
    def distance_variables_dimension(self):
        mat_dim = self.num_range_measurements + 1
        return mat_dim

    @property
    def num_range_measurements(self):
        return len(self.range_measurements)

    @property
    def num_odom_measurements(self):
        return len(self.odometry_measurements)

    @property
    def num_total_measurements(self):
        return self.num_range_measurements + self.num_odom_measurements

    @property
    def dist_measurements_vect(self) -> np.ndarray:
        """
        Get a vector of the distance measurements

        Returns:
            np.ndarray: a vector of the distance measurements
        """
        return np.array([meas.dist for meas in self.range_measurements])

    @property
    def weighted_dist_measurements_vect(self) -> np.ndarray:
        """
        Get of the distance measurements weighted by their precision
        """
        return self.dist_measurements_vect * self.measurements_weight_vect

    @property
    def measurements_weight_vect(self) -> np.ndarray:
        """
        Get the weights of the measurements
        """
        return np.array([meas.weight for meas in self.range_measurements])

    @property
    def sum_weighted_measurements_squared(self) -> float:
        """
        Get the sum of the squared weighted measurements
        """
        weighted_dist_vect = self.weighted_dist_measurements_vect
        dist_vect = self.dist_measurements_vect
        return np.dot(weighted_dist_vect, dist_vect)

    def get_range_measurement_pose(self, measure: RangeMeasurement) -> PoseVariable:
        """Gets the pose associated with the range measurement

        Arguments:
            measure (RangeMeasurement): the range measurement

        Returns:
            PoseVariable: the pose variable associated with the range
                measurement
        """
        pose_name = measure.var1
        pose_idx = int(pose_name[1:])
        return self.pose_variables[pose_idx]

    def get_range_measurement_landmark(
        self, measure: RangeMeasurement
    ) -> LandmarkVariable:
        """Returns the landmark variable associated with this range measurement

        Arguments:
            measure (RangeMeasurement): the range measurement

        Returns:
            LandmarkVariable: the landmark variable associated with the range
                measurement
        """
        landmark_name = measure.var2
        landmark_idx = int(landmark_name[1:])
        return self.landmark_variables[landmark_idx]

    def get_pose_translation_variable_indices(
        self, pose: PoseVariable
    ) -> Tuple[int, int]:
        """
        Get the indices [start, stop) for the translation variable corresponding to this pose
        in the factor graph

        Args:
            pose (PoseVariable): the pose variable

        Returns:
            Tuple[int, int]: [start, stop) the start and stop indices
        """
        assert pose in self.pose_variables
        pose_idx = self.pose_variables.index(pose)
        d = self.dimension

        # get the start and stop indices for the translation variables
        start = pose_idx * d
        stop = (pose_idx + 1) * d

        return (start, stop)

    def get_landmark_translation_variable_indices(
        self, landmark: LandmarkVariable
    ) -> Tuple[int, int]:
        """
        Get the indices [start, stop) for the translation variable corresponding to this landmark
        in the factor graph

        Args:
            landmark (LandmarkVariable): the landmark variable

        Returns:
            Tuple[int, int]: [start, stop) the start and stop indices
        """
        assert landmark in self.landmark_variables
        landmark_idx = self.landmark_variables.index(landmark)

        # offset due to the pose translations
        d = self.dimension
        offset = self.num_poses * d

        # get the start and stop indices for the translation variables
        start = landmark_idx * d + offset
        stop = start + d

        return (start, stop)

    def get_pose_rotation_variable_indices(self, pose: PoseVariable) -> Tuple[int, int]:
        """
        Get the indices [start, stop) for the rotation variable corresponding to
        this pose in the factor graph

        Args:
            pose (PoseVariable): the pose variable

        Returns:
            Tuple[int, int]: [start, stop) the start and stop indices
        """
        assert pose in self.pose_variables
        pose_idx = self.pose_variables.index(pose)
        d = self.dimension

        # need an offset to skip over all the translation variables
        offset = self.num_translations * d

        # get the start and stop indices
        start = (pose_idx * d * d) + offset
        stop = start + d * d

        return (start, stop)

    def get_range_dist_variable_indices(self, measurement: RangeMeasurement) -> int:
        """
        Get the index for the distance variable corresponding to
        this measurement in the factor graph

        Args:
            measurement (RangeMeasurement): the measurement

        Returns:
            int: the index of the distance variable
        """
        assert measurement in self.range_measurements
        measure_idx = self.range_measurements.index(measurement)
        d = self.dimension

        # need an offset to skip over all the translation and rotation
        # variables
        range_offset = self.num_translations * d
        range_offset += self.num_poses * d * d

        # get the start and stop indices
        range_idx = (measure_idx) + range_offset

        return range_idx
