#pragma once

#include <random>
#include <vector>

#include <Eigen/Dense>

#include "SESync/RelativePoseMeasurement.h"

/** This function samples 'num_samples' realizations of the von Mises
distribution on the circle with mode mu and concentration parameter kappa.  The
mode mu and the returned samples are parameterized as angles taking values in
the range [-pi, pi). Internally, this function implements the rejection sampler
describedin the paper "Efficient Simulation of the von Mises Distribution" by
Best and Fisher. */
std::vector<double> sample_von_Mises(double mu, double kappa,
                                     std::default_random_engine &generator,
                                     unsigned int num_samples = 1);

/** This function returns 'num_samples' realizations of the isotropic Langevin
 * distribution on SO(d) with mode M and concentration parameter kappa, for d =
 * 2 or 3. */
std::vector<Eigen::MatrixXd>
sample_rotational_isotropic_Langevin(const Eigen::MatrixXd &M, double kappa,
                                     std::default_random_engine &generator,
                                     unsigned int num_samples = 1);

/** This function generates a sample of the Cube dataset described in the tech
 * report */
std::vector<SESync::RelativePoseMeasurement>
generate_cube_dataset(unsigned int s, double kappa, double tau, double pLC,
                      std::default_random_engine &generator, Eigen::MatrixXd &t,
                      Eigen::MatrixXd R);
