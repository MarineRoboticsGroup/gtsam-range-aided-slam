#include "sampling.h"

#include <math.h>
#include <random>

std::vector<double> sample_von_Mises(double mu, double kappa,
                                     std::default_random_engine &generator,
                                     unsigned int num_samples) {
  // Constants defined in the original paper
  double tau = 1 + sqrt(1 + 4 * kappa * kappa);
  double rho = (tau - sqrt(2 * tau)) / (2 * kappa);
  double r = (1 + rho * rho) / (2 * rho);

  // Allocate output
  std::vector<double> theta(num_samples);

  // Set up a random sampler for the standard uniform distribution
  std::uniform_real_distribution<double> Uniform(0.0, 1.0);

  double z, f, c;
  for (unsigned int i = 0; i < num_samples; i++) {

    if (kappa > 0) {
      bool accepted = false;

      while (!accepted) {
        // Draw two random uniform numbers
        double u1 = Uniform(generator);
        double u2 = Uniform(generator);

        // Step 1 from the paper
        z = cos(M_PI * u1);
        f = (1 + r * z) / (r + z);
        c = kappa * (r - f);

        // Step 3 from the paper (Step 2 is a short-circuit optimization)
        if (log(c / u2) + 1 >= c)
          accepted = true;
      }

      // Signum function
      auto sign = [](double x) { return (x < 0) ? -1 : (x > 0); };

      // This accepted value is a sample drawn from the von Mises distribution
      // vM(mu, kappa), but possibly lying outside the range (-pi, pi].
      theta[i] = sign(Uniform(generator) - .5) * acos(f) + mu;

      // Remap theta to (-pi, pi] if necessary
      if (theta[i] > M_PI)
        theta[i] -= M_PI;
      if (theta[i] < -M_PI)
        theta[i] += M_PI;

    } // if (kappa > 0)
    else {
      // kappa = 0 just corresponds to sampling from a uniform distribution on
      // the circle
      theta[i] = 2 * M_PI * (Uniform(generator) - .5);
    }
  } // for( ... )
  return theta;
}

std::vector<Eigen::MatrixXd>
sample_rotational_isotropic_Langevin(const Eigen::MatrixXd &M, double kappa,
                                     std::default_random_engine &generator,
                                     unsigned int num_samples) {
  unsigned int d = M.rows();

  // The sampling procedure outlined below makes use of the fact that the
  // distribution over the rotation angle of the relative rotation M^-1 R
  // induced by Langevin(M, kappa) is vM(0, 2*kappa), the von Mises distribution
  // on the circle with mode 0 and concentration parameter kappa

  std::vector<double> theta =
      sample_von_Mises(0, 2 * kappa, generator, num_samples);

  // Set up a random sampler for the standard Gaussian distribution
  std::normal_distribution<double> gaussian(0.0, 1.0);

  std::vector<Eigen::MatrixXd> R(num_samples);

  for (unsigned i = 0; i < num_samples; i++)
    if (d == 2) {

      // Just convert the sampled angle into a 2D rotation matrix
      Eigen::Matrix2d rot;
      rot << cos(theta[i]), sin(theta[i]), -sin(theta[i]), cos(theta[i]);

      R[i] = rot;
    } else if (d == 3) {
      // In 3 dimensions, Langevin(M, kappa) is gotten by perturbing M with a
      // rotation whose *absolute magnitude* is given by abs(theta), and whose
      // axis of rotation is uniformly distributed over S^2

      // Sample an axis of rotation uniformly over S^2, exploiting the fact that
      // a vector of iid random Gaussians has an isotropic distribution

      Eigen::Vector3d vhat;
      vhat(0) = gaussian(generator);
      vhat(1) = gaussian(generator);
      vhat(2) = gaussian(generator);
      vhat.normalize();

      Eigen::Matrix3d V;
      V << 0, -vhat(2), vhat(1), vhat(2), 0, -vhat(0), -vhat(1), vhat(0), 0;

      double omega = fabs(theta[i]);
      Eigen::Matrix3d dRk = Eigen::Matrix3d::Identity() + sin(omega) * V +
                            (1 - cos(omega)) * V * V;

      R[i] = M * dRk;
    }

  return R;
}

std::vector<SESync::RelativePoseMeasurement>
generate_cube_dataset(unsigned int s, double kappa, double tau, double pLC,
                      std::default_random_engine &generator, Eigen::MatrixXd &t,
                      Eigen::MatrixXd R) {

  Eigen::Matrix3d I3 = Eigen::Matrix3d::Identity();

  // Set up random number generation
  std::normal_distribution<double> gaussian(0.0, 1.0);
  std::uniform_real_distribution<double> uniform(0.0, 1.0);

  unsigned int num_poses = s * s * s;

  R.resize(3, 3 * num_poses);
  t.resize(3, num_poses);

  // Allocate vector of pose-graph measurements
  std::vector<SESync::RelativePoseMeasurement> measurements;

  /// Construct ground-truth poses
  unsigned int pose_id = 0;
  for (unsigned int l = 0; l < s; l++)
    for (unsigned int j = 0; j < s; j++)
      for (unsigned int i = 0; i < s; i++) {
        // Sample a uniformly random orientation
        R.block<3, 3>(0, 3 * pose_id) =
            sample_rotational_isotropic_Langevin(I3, 0.1, generator).front();

        if (j % 2 == 0) {
          // We move up this column from bottom to top
          t(0, pose_id) = j;
          t(1, pose_id) = i;
          t(2, pose_id) = l;
        } else {
          // We move down this column from the top
          t(0, pose_id) = j;
          t(1, pose_id) = s - 1 - i;
          t(2, pose_id) = l;
        }

        ++pose_id;
      }

  /// Construct odometric edges

  /// Helper function to compute relative pose measurements
  auto compute_relative_pose =
      [](const Eigen::Vector3d &ti, const Eigen::Matrix3d &Ri,
         const Eigen::Vector3d &tj, const Eigen::Matrix3d &Rj,
         Eigen::Vector3d &tij, Eigen::Matrix3d &Rij) {

        tij = Ri.transpose() * (tj - ti);
        Rij = Ri.transpose() * Rj;
      };

  /// Helper function to generate noisy relative pose measurements
  auto compute_relative_pose_observation =
      [&compute_relative_pose, &gaussian, &generator, &t, &R, kappa,
       &tau](unsigned int i, unsigned int j) {

        // Compute ground-truth relative measurements
        Eigen::Vector3d tij;
        Eigen::Matrix3d Rij;
        compute_relative_pose(t.col(i), R.block<3, 3>(0, 3 * i), t.col(j),
                              R.block<3, 3>(0, 3 * j), tij, Rij);

        // Add noise
        for (unsigned int k = 0; k < 3; k++)
          tij(k) += (1 / sqrt(tau)) * gaussian(generator);

        Rij = Rij *
              sample_rotational_isotropic_Langevin(Eigen::Matrix3d::Identity(),
                                                   kappa, generator)
                  .front();

        SESync::RelativePoseMeasurement obs;
        obs.i = i;
        obs.j = j;
        obs.kappa = kappa;
        obs.tau = tau;
        obs.t = tij;
        obs.R = Rij;

        return obs;

      };

  /// Compute odometric measurements
  for (unsigned int i = 0; i < num_poses - 1; i++)
    measurements.push_back(compute_relative_pose_observation(i, i + 1));

  /// Create loop closures
  for (unsigned int i = 0; i < num_poses; i++) {
    for (unsigned int j = i + 2; j < num_poses; j++) {

      // If poses i and j are nearby (and non-consecutive), sample a loop
      // closure with probability p
      if ((t.col(i) - t.col(j)).norm() <= 1.5) {
        // Sample this loop closure observation with probability p
        if (uniform(generator) < pLC)
          measurements.emplace_back(compute_relative_pose_observation(i, j));
      }
    }
  }

  return measurements;
}
