#include "SESyncFactor3d.h"

Eigen::Matrix3d cross_product_matrix(const Eigen::Vector3d& omega)
{
    Eigen::Matrix3d S;
    S << 0, -omega(2), omega(1),
         omega(2), 0, -omega(0),
         -omega(1), omega(0), 0;

            return S;
}

gtsam::Vector SESyncFactor3d::evaluateError(const gtsam::Pose3& Xi, const gtsam::Pose3& Xj, boost::optional<gtsam::Matrix&> H1, boost::optional<gtsam::Matrix&> H2) const
{
    gtsam::Vector residual(12);

    // First, compute the translational component of the residual vector
    residual.head<3>() = (Xj.translation() - Xi.translation() - Xi.rotation() * tij);

    // Next, compute the rotational component of the residual vector
    gtsam::Matrix Rerr_mat = Xj.rotation().matrix() - (Xi.rotation() * Rij).matrix();
    residual.tail<9>() = Eigen::Map<Eigen::VectorXd>(Rerr_mat.data(), 9);


    /** For derivative computations, note that GTSAM considers Pose3 tangent vectors to be laid out as (omega, v) in R^6, where omega is angular velocity and v translational velocity).
     *
     * Therefore, the Jacobian of the residual vector
     *
     * r = [tj - ti - Ri tij;
     *      vec(Rj - Ri * Rij)]
     *
     * has the block structure
     *
     * J = [ dt_err / dR, dt_err / dt ;
     *       dvec(R_err) / dR, 0];
     */

    if(H1)
    {
        // Compute the Jacobian of the residual vector wrt pose Xi.  Note that this will be a 12 x 6 matrix, since the residual vector is 12-dimensional, but the intrinsic dimension of Xi is 6.
        *H1 = Eigen::MatrixXd::Zero(12, 6);

        /** Set dt_err / dR_i
         *
         * Note that since t_err = t_j - t_i - R_i t_ij is linear in R_i, then
         *
         * dt_err / dw = - d / dw [R_i exp([w]_x)] t_ij = - R_i (d / dw) ( [w]_x t_ij ) =  R_i d/dw ([t_ij]_x w) = R_i [t_ij]_x
         *
         */

        Eigen::MatrixXd Ri = Xi.rotation().matrix();
        H1->topLeftCorner<3,3>() = Ri * cross_product_matrix(tij);

        // Set dt_err / dt_i
        H1->topRightCorner<3,3>() = -Ri;



        // Set dvec(R_err) / dR
        Eigen::Matrix<double,9,3> dvecRerr_dRi = Eigen::Matrix<double,9,3>::Zero();

        for(unsigned int c = 0; c < 3; c++)
            for(unsigned int r = 0; r < 3; r++)
                dvecRerr_dRi.row(3*c + r) = -Rij.column(c + 1).cross(Ri.row(r));

        H1->bottomLeftCorner<9,3>() = dvecRerr_dRi;

        // dvec(R_err) / dt_i = 0
    }

    if(H2)
    {
        // Compute the Jacobian of the residual vector wrt pose Xi.  Note that this will be a 12 x 6 matrix, since the residual vector is 12-dimensional, but the intrinsic dimension of Xi is 6.
        *H2 = Eigen::MatrixXd::Zero(12, 6);

        Eigen::MatrixXd Rj = Xj.rotation().matrix();

        // dt_err / dRj = 0

        // Set dt_err / dtj
        H2->topRightCorner<3,3>() = Rj;


        // Set dvec(R_err) / dRj
        H2->block<3,1>(3, 1) = -Rj.col(2);
        H2->block<3,1>(3, 2) = Rj.col(1);

        H2->block<3,1>(6, 0) = Rj.col(2);
        H2->block<3,1>(6, 2) = -Rj.col(0);

        H2->block<3,1>(9, 0) = -Rj.col(1);
        H2->block<3,1>(9, 1) = Rj.col(0);

        // dvec(R_err) / dt = 0
    }

    return residual;
}
