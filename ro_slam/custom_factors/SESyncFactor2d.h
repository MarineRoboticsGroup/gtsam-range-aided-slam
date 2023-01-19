#ifndef _SESYNCFACTOR2D_H_
#define _SESYNCFACTOR2D_H_

#include <Eigen/Dense>

#include <gtsam/geometry/Pose2.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

/** This class implements a relative pose measurement loss function of the form
 *
 * e_ij = \kappa_ij * || R_j - R_i * R_ij ||_F^2 + \tau_ij * || t_j - t_i - R_i
 * * t_ij ||_2^2
 *
 */

class SESyncFactor2d
    : public gtsam::NoiseModelFactor2<gtsam::Pose2, gtsam::Pose2> {
 public:
  /** Superclass */
  typedef gtsam::NoiseModelFactor2<gtsam::Pose2, gtsam::Pose2> Super;

 private:
  /** Raw measurements and measurement precisions */
  double kappa;
  double tau;
  gtsam::Rot2 Rij;
  gtsam::Point2 tij;

 public:
  // Constructor
  SESyncFactor2d(gtsam::Key key_i, gtsam::Key key_j,
                 const gtsam::Rot2& relative_rotation,
                 const gtsam::Point2& relative_translation,
                 double rotational_precision, double translational_precision)
      : Super(gtsam::noiseModel::Diagonal::Precisions(
                  (gtsam::Vector(6) << 2 * translational_precision,
                   2 * translational_precision, 2 * rotational_precision,
                   2 * rotational_precision, 2 * rotational_precision,
                   2 * rotational_precision)
                      .finished()),
              key_i, key_j),
        kappa(rotational_precision),
        tau(translational_precision),
        Rij(relative_rotation),
        tij(relative_translation) {}

  /// Overridden pure virtual function from the super class; computes the raw
  /// measurement residual vector
  gtsam::Vector evaluateError(
      const gtsam::Pose2& Xi, const gtsam::Pose2& Xj,
      boost::optional<gtsam::Matrix&> H1 = boost::none,
      boost::optional<gtsam::Matrix&> H2 = boost::none) const;

  // Nothing to do here
  ~SESyncFactor2d() {}
};

/// Helper functions

// Given an angular velocity omega, this function computes and returns the
// associated element of the (1-dimensional) Lie algebra so(2)
Eigen::Matrix2d tangent_matrix(double omega);

#endif  // _SESYNCFACTOR2D_H_
