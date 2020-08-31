#ifndef FusionEKF_H_
#define FusionEKF_H_

#include "measurement_package.h"
#include "Eigen/Dense"
#include <vector>
#include <string>
#include <fstream>
#include "kalman_filter.h"
#include "tools.h"

class FusionEKF {
public:
  /**
  * Constructor.
  */
  FusionEKF();

  /**
  * Destructor.
  */
  virtual ~FusionEKF();

  /**
  * Run the whole flow of the Kalman Filter from here.
  */
  void ProcessMeasurement(const MeasurementPackage &measurement_pack);

  /**
  * Kalman Filter update and prediction math lives in here.
  */
  KalmanFilter ekf_;

private:
  // check whether the tracking toolbox was initiallized or not (first measurement)
  bool is_initialized_;

  // previous timestamp
  long long previous_timestamp_;

  Eigen::MatrixXd R_laser_;  // Noise Covariance Matrix for Laser
  Eigen::MatrixXd R_radar_;  // Noise Covariance Matrix for Radar
  Eigen::MatrixXd H_laser_;  // Observation model H
                             // (maps true state space into observed state space)
  Eigen::MatrixXd Hj_;  // Jacobian H
  Eigen::VectorXd h(Eigen::VectorXd &x); // Converts 4x1 cartesian into 3x1 radial
  float noise_ax, noise_ay;

};

#endif /* FusionEKF_H_ */
