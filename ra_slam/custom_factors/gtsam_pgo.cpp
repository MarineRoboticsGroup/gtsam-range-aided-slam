#include "gtsam_pgo.h"

#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/slam/PriorFactor.h>

#include "SESyncFactor2d.h"
#include "SESyncFactor3d.h"

#include "Optimization/Util/Stopwatch.h"
#include "SESync/SESyncProblem.h"

#include <iostream>

GTSAMResult
gtsam_pgo(const std::vector<SESync::RelativePoseMeasurement> &measurements,
          const gtsam::NonlinearOptimizerParams &params,
          const OptimizerType &opt_type) {

  std::cout << "GTSAM POSE-GRAPH OPTIMIZATION" << std::endl << std::endl;

  // Get the dimension of the problem that we'll be solving
  unsigned int dim = measurements[0].t.size();

  /// CONSTRUCT FACTOR GRAPH

  std::cout << "Constructing factor graph ... " << std::endl;
  gtsam::NonlinearFactorGraph graph;

  if (dim == 3) {
    // We're solving a 3D problem
    /// Add factors for each measurement
    for (const SESync::RelativePoseMeasurement &m : measurements)
      graph.add(SESyncFactor3d(m.i, m.j, gtsam::Rot3(m.R), gtsam::Point3(m.t),
                               m.kappa, m.tau));

    /// Add a prior constraining the first pose to zero
    graph.add(gtsam::PriorFactor<gtsam::Pose3>(
        0, gtsam::Pose3(),
        gtsam::noiseModel::Isotropic::Sigma(gtsam::Pose3::dimension, 100)));
  } else if (dim == 2) {
    // We're solving a 2D problem
    /// Add factors for each measurement
    for (const SESync::RelativePoseMeasurement &m : measurements)
      graph.add(
        SESyncFactor2d(m.i, m.j,
                               gtsam::Rot2::fromCosSin(m.R(0, 0), m.R(1, 0)),
                               gtsam::Point2(m.t), m.kappa, m.tau)
                               );

    /// Add a prior constraining the first pose to zero
    graph.add(gtsam::PriorFactor<gtsam::Pose2>(
        0, gtsam::Pose2(),
        gtsam::noiseModel::Isotropic::Sigma(gtsam::Pose2::dimension, 100)));
  } else {
    std::cout << "Error: Can't solve a problem in dimension " << dim << "!"
              << std::endl;
    exit(1);
  }

  /// COMPUTE CHORDAL INITIALIZATION
  std::cout << "Computing chordal initialization ... " << std::endl;
  SESync::SESyncProblem problem(measurements, SESync::Formulation::Explicit);
  problem.set_relaxation_rank(dim);
  SESync::Matrix X0 = problem.chordal_initialization();
  SESync::Matrix t0 = X0.leftCols(problem.num_poses());
  SESync::Matrix R0 = X0.rightCols(problem.num_poses() * problem.dimension());

  gtsam::Values initial_values;

  if (dim == 3) {
    // We insert 3D pose estimates
    for (unsigned int i = 0; i < t0.cols(); i++)
      initial_values.insert(i,
                            gtsam::Pose3(gtsam::Rot3(R0.block<3, 3>(0, 3 * i)),
                                         gtsam::Point3(t0.col(i))));
  } else {
    // We insert 2d pose estimates
    for (unsigned int i = 0; i < t0.cols(); i++)
      initial_values.insert(
          i, gtsam::Pose2(gtsam::Rot2::fromCosSin(R0(0, 2 * i), R0(1, 2 * i)),
                          gtsam::Point2(t0.col(i))));
  }

  /// CONSTRUCT OPTIMIZER

  gtsam::NonlinearOptimizer *optimizer;
  if (opt_type == OptimizerType::DogLeg) {
    std::cout << "Constructing dog-leg optimizer ... " << std::endl
              << std::endl;

    const gtsam::DoglegParams &dogleg_params =
        static_cast<const gtsam::DoglegParams &>(params);
    optimizer =
        new gtsam::DoglegOptimizer(graph, initial_values, dogleg_params);
    ;
  } else // OptimizerType == LM
  {
    std::cout << "Constructing Levenberg-Marquardt optimizer ... " << std::endl
              << std::endl;
    const gtsam::LevenbergMarquardtParams &lm_params =
        static_cast<const gtsam::LevenbergMarquardtParams &>(params);
    optimizer = new gtsam::LevenbergMarquardtOptimizer(graph, initial_values,
                                                       lm_params);
  }

  auto start_time = Stopwatch::tick();
  optimizer->optimize();
  double elapsed_time = Stopwatch::tock(start_time);

  std::cout << std::endl << std::endl;

  GTSAMResult result;
  result.optimization_time = elapsed_time;
  result.objective_value = optimizer->error();

  delete optimizer;

  return result;
}
