#pragma once

#include <vector>

#include <gtsam/nonlinear/DoglegOptimizer.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearOptimizer.h>

#include "SESync/RelativePoseMeasurement.h"

// The type of optimizer to use
enum class OptimizerType { DogLeg, LM };

struct GTSAMResult {
  double objective_value;
  double optimization_time;
};

GTSAMResult
gtsam_pgo(const std::vector<SESync::RelativePoseMeasurement> &measurements,
          const gtsam::NonlinearOptimizerParams &params,
          const OptimizerType &optimizer = OptimizerType::DogLeg);
