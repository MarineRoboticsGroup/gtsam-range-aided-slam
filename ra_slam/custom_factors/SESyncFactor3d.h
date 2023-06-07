#ifndef _SESYNCFACTOR3D_H_
#define _SESYNCFACTOR3D_H_

#include <Eigen/Dense>

#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/NonlinearFactor.h>

/** This class implements a relative pose measurement loss function of the form
 *
 * e_ij = \kappa_ij * || R_j - R_i * R_ij ||_F^2 + \tau_ij * || t_j - t_i - R_i
 * * t_ij ||_2^2
 *
 */

class SESyncFactor3d
    : public gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Pose3> {
 public:
  /** Superclass */
  typedef gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Pose3> Super;

 private:
  /** Raw measurements and measurement precisions */
  double kappa;
  double tau;
  gtsam::Rot3 Rij;
  gtsam::Point3 tij;

 public:
  // Constructor
  SESyncFactor3d(gtsam::Key key_i, gtsam::Key key_j,
                 const gtsam::Rot3& relative_rotation,
                 const gtsam::Point3& relative_translation,
                 double rotational_precision, double translational_precision)
      : Super(gtsam::noiseModel::Diagonal::Precisions(
                  (gtsam::Vector(12) << 2 * translational_precision,
                   2 * translational_precision, 2 * translational_precision,
                   2 * rotational_precision, 2 * rotational_precision,
                   2 * rotational_precision, 2 * rotational_precision,
                   2 * rotational_precision, 2 * rotational_precision,
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
      const gtsam::Pose3& Xi, const gtsam::Pose3& Xj,
      boost::optional<gtsam::Matrix&> H1 = boost::none,
      boost::optional<gtsam::Matrix&> H2 = boost::none) const;

  // Nothing to do here
  ~SESyncFactor3d() {}
};

/// Helper functions

// Given a 3-vector omega, this computes and returns the cross product matrix
// [w]_x
Eigen::Matrix3d cross_product_matrix(const Eigen::Vector3d& omega);

#endif  // _SESYNCFACTOR3D_H_
