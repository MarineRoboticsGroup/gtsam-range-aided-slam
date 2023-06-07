#include "SESyncFactor2d.h"

Eigen::Matrix2d tangent_matrix(double omega) {
  Eigen::Matrix2d Omega;
  Omega << 0, -omega, omega, 0;
  return Omega;
}

gtsam::Vector SESyncFactor2d::evaluateError(
    const gtsam::Pose2& Xi, const gtsam::Pose2& Xj,
    boost::optional<gtsam::Matrix&> H1,
    boost::optional<gtsam::Matrix&> H2) const {
  gtsam::Vector residual(6);

  // First, compute the translational component of the residual vector
  residual.head<2>() =
      (Xj.translation() - Xi.translation() - Xi.rotation() * tij);

  // Next, compute the rotational component of the residual vector
  gtsam::Matrix Rerr_mat =
      Xj.rotation().matrix() - (Xi.rotation() * Rij).matrix();
  residual.tail<4>() = Eigen::Map<Eigen::VectorXd>(Rerr_mat.data(), 4);

  /** For derivative computations, note that GTSAM considers Pose2 tangent
   * vectors to be laid out as (v, omega) in R^3, where omega is angular
   * velocity and v translational velocity).
   *
   * Therefore, the Jacobian of the residual vector
   *
   * r = [tj - ti - Ri tij;
   *      vec(Rj - Ri * Rij)]
   *
   * has the block structure
   *
   * J = [ dt_err / dt, dt_err / dR ;
   *       0, dvec(R_err) / dR];
   */

  if (H1) {
    // Compute the Jacobian of the residual vector wrt pose Xi.  Note that this
    // will be a 6 x 3 matrix, since the residual vector is 6-dimensional, but
    // the intrinsic dimension of Xi is 3.
    *H1 = Eigen::MatrixXd::Zero(6, 3);

    Eigen::Matrix2d Ri = Xi.rotation().matrix();
    Eigen::Matrix2d dRi_domegai = tangent_matrix(1);

    // Set dt_err / dt_i
    H1->topLeftCorner<2, 2>() = -Ri;

    // Set dt_err / dR_i
    H1->topRightCorner<2, 1>() = -Ri * dRi_domegai * tij;

    // dR_err / dt_i = 0

    // Set dR_err / dR_i

    Eigen::Matrix2d dRerr_dRi = -Ri * dRi_domegai * Rij.matrix();

    H1->bottomRightCorner<4, 1>() =
        Eigen::Map<Eigen::Vector4d>(dRerr_dRi.data());
  }

  if (H2) {
    // Compute the Jacobian of the residual vector wrt pose Xj.  Note that this
    // will be a 6 x 3 matrix, since the residual vector is 6-dimensional, but
    // the intrinsic dimension of Xi is 3.
    *H2 = Eigen::MatrixXd::Zero(6, 3);

    Eigen::Matrix2d Rj = Xj.rotation().matrix();

    // Set dt_err / dt_j
    H2->topLeftCorner<2, 2>() = Rj;

    // dt_err / dR_j = 0

    // Set dR_err / dt_j = 0

    // dR_err / dR_j

    Eigen::Matrix2d dRerr_dRj = Rj * tangent_matrix(1);

    H2->bottomRightCorner<4, 1>() =
        Eigen::Map<Eigen::Vector4d>(dRerr_dRj.data());
  }

  return residual;
}
