#include "kalman_filter.h"
#include <cmath>
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::cout;
using std::endl;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

// This is the prediction step.
// Make sure to set F and Q before doing this.
void KalmanFilter::Predict() {
  x_ = F_ * x_;
  P_ = (F_ * P_ * F_.transpose()) + Q_;

}

// This is the update function for a lidar.
// Make sure to set P, H, and R before doing this.
void KalmanFilter::Update(const VectorXd &z) {

  MatrixXd Ht = H_.transpose();

  VectorXd y = z - H_ * x_;
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd K = P_ * Ht * S.inverse();

  // New Estimates
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;

}


// This is the update function for a radar.
// Make sure to set P, H, and R before doing this.
void KalmanFilter::UpdateEKF(const VectorXd &z) {

  MatrixXd Ht = H_.transpose();
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);

  // Calculations
  VectorXd y = z - Tools::CartesianToRadial(x_);
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd K = P_ * Ht * S.inverse();

  // New Estimates
  x_ = x_ + K * y;
  P_ = (I - K * H_) * P_;

}

