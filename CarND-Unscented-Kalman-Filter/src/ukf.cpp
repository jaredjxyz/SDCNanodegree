#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  ///* initially set to false, set to true in first call of ProcessMeasurement
  is_initialized_ = false;

  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  ///* State dimension
  n_x_ = 5;

  ///* Augmented state dimension
  n_aug_ = n_x_ + 2;  // Length of state plus length of number of process noise variables

  // initial state vector
  x_ = VectorXd(n_x_);

  // initial covariance matrix
  P_ = MatrixXd(n_x_, n_x_);

  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  // This is relatively arbitrary and could be changed
  std_a_ = .5;

  // Process noise standard deviation yaw acceleration in rad/s^2
  // This is relatively arbitrary and could be changed
  std_yawdd_ = .5;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  ///* Sigma point spreading parameter
  // This is arbitrary and can be changed.
  lambda_ = 2.4 - n_aug_;

  ///* Weights of sigma points
  // TODO: Set this to correct initial value
  weights_ = VectorXd::Constant(2 * n_aug_ + 1, 1 / (2 * (lambda_ + n_aug_)));
  weights_(0) = lambda_ / (lambda_ + n_aug_);

  ///* the current NIS for radar
  // TODO: Set this after setting up the NIS
  NIS_radar_;

  ///* the current NIS for laser
  NIS_laser_;

  //Hint: one or more values initialized above might be wildly off...
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Make sure you switch between lidar and radar
  measurements.
  */
  if (!is_initialized_) {
  // first measurement
    previous_timestamp = meas_package.timestamp_;

    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {

      float rho = meas_package.raw_measurements_[0];
      float gamma = meas_package.raw_measurements_[1];
      float rho_dot = meas_package.raw_measurements_[2];

      //State consists of 5 elements
      float p_x = rho * cos(gamma);
      float p_y = rho * sin(gamma);
      float v = rho_dot;
      float psi = gamma;
      float psi_dot = 0;

      x_ << p_x, p_y, v, psi, psi_dot;

      // We know everything pretty well except for angular acceleration
      P_ << 1, 0, 0, 0, 0,
            0, 1, 0, 0, 0,
            0, 0, 1, 0, 0,
            0, 0, 0, 1, 0,
            0, 0, 0, 0, 1;

    } else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {

      float p_x = meas_package.raw_measurements_[0];
      float p_y = meas_package.raw_measurements_[1];
      float v = 0;
      float psi = (p_x == 0 && p_y == 0) ? 0 : atan2(p_y, p_x);
      float psi_dot = 0;

      x_ << p_x, p_y, v, psi, psi_dot;

      // We only know p_x, p_y and psi well
      P_ << .1, 0, 0, 0, 0,
            0, .1, 0, 0, 0,
            0, 0, .01, 0, 0,
            0, 0, 0, 1, 0,
            0, 0, 0, 0, 1;
    }

    is_initialized_ = true;
    return;
  }

  // Convert delta t to seconds from us
  double delta_t = (double)(meas_package.timestamp_ - previous_timestamp) / 1000000.0;
  cout << "new time " << meas_package.timestamp_ << endl;
  cout << "old time " << previous_timestamp << endl;
  cout << "diff" << ((double)(meas_package.timestamp_ - previous_timestamp) / 1000000.0 )<< endl;

  // Only predict if enough time has passed
  cout << meas_package.raw_measurements_;
  Prediction(delta_t);
  previous_timestamp = meas_package.timestamp_;


  if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
    UpdateRadar(meas_package);

  } else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
    UpdateLidar(meas_package);
  }

}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {

  // Generate Sigma Points

  // Setup matrices
  MatrixXd A = P_.llt().matrixL();  // Square root of P
  MatrixXd Xsig = MatrixXd(n_x_, 2 * n_x_ + 1);  // Sigma points

  // Get A size

  // Xsig = [x, sqrt(lambda + nx) * A + x, -sqrt(lambda + nx) * A + x]
  Xsig.col(0) = x_;
  Xsig.block(0, 1, n_x_, n_x_) =   (sqrt(lambda_ + n_x_) * A).colwise() + x_;
  Xsig.block(0, 1 + n_x_, n_x_, n_x_) =  (-sqrt(lambda_ + n_x_) * A).colwise() + x_;

  // Predict Sigma Points

  // Build the augmented state
  VectorXd x_aug = VectorXd(n_aug_);
  x_aug << x_, 0, 0;

  // Build augmented covariance matrix
  // P_aug = [[P, 0],
  //          [0, Q]
  MatrixXd Q = MatrixXd(2, 2);
  Q << std_a_ * std_a_, 0,
       0, std_yawdd_ * std_yawdd_;

  MatrixXd P_aug = MatrixXd::Zero(n_aug_, n_aug_);
  P_aug.block(0, 0, n_x_, n_x_) = P_;
  P_aug.block(n_x_, n_x_, 2, 2) = Q;

  // Augment sigma points
  MatrixXd A_aug = P_aug.llt().matrixL();  // Square root of P_aug
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);
  Xsig_aug.col(0) = x_aug;
  Xsig_aug.block(0, 1, n_aug_, n_aug_) =   (sqrt(lambda_+ n_aug_) * A_aug).colwise() + x_aug;
  Xsig_aug.block(0, 1 + n_aug_, n_aug_, n_aug_) =  (-sqrt(lambda_ + n_aug_) * A_aug).colwise() + x_aug;

  // Predict new sigma points one by one
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    double p_x = Xsig_aug(0, i);
    double p_y = Xsig_aug(1, i);
    double v = Xsig_aug(2, i);
    double psi = Xsig_aug(3, i);
    double psi_dot = Xsig_aug(4, i);
    double v_a = Xsig_aug(5, i);
    double v_psi_dot_dot = Xsig_aug(6, i);

    //Avoid division by zero
    if (psi_dot == 0) {
        Xsig_pred_(0, i) = p_x + v * cos(psi) * delta_t
                          + .5 * delta_t * delta_t * cos(psi) * v_a;
        Xsig_pred_(1, i) = p_y + v * sin(psi) * delta_t
                          + .5 * delta_t * delta_t * sin(psi) * v_a;
        Xsig_pred_(2, i) = v + delta_t * v_a;
        Xsig_pred_(3, i) = psi + psi_dot * delta_t
                          + .5 * delta_t * delta_t * v_psi_dot_dot;
        Xsig_pred_(4, i) = psi_dot + delta_t * v_psi_dot_dot;

    } else {
        Xsig_pred_(0, i) = p_x + (v/psi_dot) * (sin(psi + psi_dot * delta_t) - sin(psi))
                          + .5 * delta_t * delta_t * cos(psi) * v_a;
        Xsig_pred_(1, i) = p_y + (v/psi_dot) * (-cos(psi + psi_dot * delta_t) + cos(psi))
                          + .5 * delta_t * delta_t * sin(psi) * v_a;
        Xsig_pred_(2, i) = v + delta_t * v_a;
        Xsig_pred_(3, i) = psi + psi_dot * delta_t
                          + .5 * delta_t * delta_t * v_psi_dot_dot;
        Xsig_pred_(4, i) = psi_dot + v_psi_dot_dot * delta_t;
    }
  }

  // Predict the mean
  x_ = Xsig_pred_ * weights_;

  // Predict the covariance
  MatrixXd x_diff = Xsig_pred_.colwise() - x_;
  P_ = x_diff  * weights_.asDiagonal() * x_diff.transpose();

  cout << "x pred" << endl << x_ << endl;
  cout << "P pred" << endl << P_ << endl;
}



/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */

  // TODO: Predict measurement

  // Transform sigma points into the measurement space
  MatrixXd Zsig = Xsig_pred_.block(0, 0, 2, 2 * n_aug_ + 1);

  // Calculate mean predicted measurement
  VectorXd z_pred = Zsig * weights_;

  // Calculate mean predicted covariance
  MatrixXd z_diff = Zsig.colwise() - z_pred;
  MatrixXd R = MatrixXd::Zero(2, 2);  // Measurement error
  R.diagonal() << std_laspx_ * std_laspx_, std_laspy_ * std_laspy_;
  MatrixXd S = z_diff * weights_.asDiagonal() * z_diff.transpose() + R;

  // Update State

  // Calculate Cross-Correlation
  VectorXd z = meas_package.raw_measurements_;
  MatrixXd Tc = (Xsig_pred_.colwise() - x_) * weights_.asDiagonal() * (Zsig.colwise() - z).transpose();

  // Calculate Kalman gain
  MatrixXd K = Tc * S.inverse();

  // Update state mean and covariance matrix
  x_ += K * (z - z_pred);
  P_ -= K * S * K.transpose();

  cout << "x lidar" << endl << x_ << endl;
  cout << "P lidar" << endl << P_ << endl;
  // TODO: Calculate Lidar NIS

}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  // Predict Measurement

  // Transform sigma points into the measurement space
  MatrixXd Zsig = MatrixXd(3, 2 * n_aug_ + 1);
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {
    double p_x = Xsig_pred_(0, i);
    double p_y = Xsig_pred_(1, i);
    double v = Xsig_pred_(2, i);
    double psi = Xsig_pred_(3, i);

    double rho = sqrt(p_x * p_x + p_y * p_y);
    double gamma = (p_y == 0 && p_x == 0) ? 0 : atan2(p_y, p_x);
    double rho_dot = (rho == 0) ? 0 : (p_x * cos(psi) * v + p_y * sin(psi) * v) / rho;

    Zsig.col(i) << rho, gamma, rho_dot;
  }

  // Calculate mean predicted measurement
  VectorXd z_pred = Zsig * weights_;

  // Calculate predicted covariance
  MatrixXd z_diff = Zsig.colwise() - z_pred;
  MatrixXd R = MatrixXd::Zero(3, 3);  // Measurement error
  R.diagonal() << std_radr_ * std_radr_, std_radphi_ * std_radphi_, std_radrd_ * std_radrd_;
  MatrixXd S = z_diff * weights_.asDiagonal() * z_diff.transpose() + R;

  //Update State

  // Calculate Cross-Correlation
  VectorXd z = meas_package.raw_measurements_;
  MatrixXd Tc = (Xsig_pred_.colwise() - x_) * weights_.asDiagonal() * (Zsig.colwise() - z).transpose();

  // Calculate Kalman gain
  MatrixXd K = Tc * S.inverse();

  // Update state mean and covariance matrix
  x_ += K * (z - z_pred);
  P_ -= K * S * K.transpose();

  // TODO: Calculate Radar NIS
  cout << "x radar" << endl << x_ << endl;
  cout << "P radar" << endl << P_ << endl;


}
