"""An example of 2 poses with an odometry factor between them."""
from gtsam.gtsam import (
    NonlinearFactorGraph,
    ISAM2Params,
    ISAM2DoglegParams,
    ISAM2,
    Values,
    Pose2,
    PriorFactorPose2,
    BetweenFactorPose2,
    noiseModel,
    symbol,
)

if __name__ == "__main__":
    fg = NonlinearFactorGraph()

    # make pose X1
    pose1 = Pose2(0.0, 0.0, 0.0)
    pose_1_symbol = symbol('x', 1)

    # make odometry
    odom_noise = noiseModel.Diagonal.Sigmas([0.2, 0.2, 0.1])
    delta_x = 2.0
    odom_rel_pose = Pose2(delta_x, 0.0, 0.0)
    odom_factor = BetweenFactorPose2(symbol('x', 1), symbol('x', 2), odom_rel_pose, odom_noise)

    # make pose X2
    pose2 = Pose2(pose1.x() + delta_x, pose1.y(), pose1.theta())
    pose_2_symbol = symbol('x', 2)

    # make prior on pose X1
    prior_noise = noiseModel.Diagonal.Sigmas([0.3, 0.3, 0.1])
    prior_factor = PriorFactorPose2(pose_1_symbol, pose1, prior_noise)

    # add factors to graph
    fg.add(prior_factor)
    fg.add(odom_factor)

    # initialize values
    initial_values = Values()
    initial_values.insert(pose_1_symbol, pose1)
    initial_values.insert(pose_2_symbol, pose2)

    # initialize ISAM2
    parameters = ISAM2Params()
    parameters.setOptimizationParams(ISAM2DoglegParams())
    isam2 = ISAM2(parameters)

    # perform optimization
    isam2.update(fg, initial_values)
    result = isam2.calculateEstimate()
    print(result.atPose2(pose_1_symbol))
    print(result.atPose2(pose_2_symbol))
