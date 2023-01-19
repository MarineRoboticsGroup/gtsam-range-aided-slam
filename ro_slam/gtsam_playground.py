from gtsam.gtsam import (
    NonlinearFactorGraph,
    ISAM2Params,
    ISAM2DoglegParams,
    LevenbergMarquardtParams,
    LevenbergMarquardtOptimizer,
    ISAM2,
    Values,
)

parameters = ISAM2Params()
dogleg_params = ISAM2DoglegParams()
dogleg_params.setWildfireThreshold(1e-10)
parameters.setOptimizationParams(dogleg_params)
parameters.setFactorization("QR")
isam_solver = ISAM2(parameters)

# initialize a L-M optimizer
lm_params = LevenbergMarquardtParams()
lm_params.setVerbosityLM("SUMMARY")
lm_params.setAbsoluteErrorTol(1e-13)
lm_params.setRelativeErrorTol(1e-13)
lm_params.setMaxIterations(1000)
lm_params.setDiagonalDamping(True)
