#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
	VectorXd rmse(4);
	rmse << 0,0,0,0;

    // TODO: YOUR CODE HERE

	// check the validity of the following inputs:
	//  * the estimation vector size should not be zero
	if(estimations.size() != ground_truth.size()) {
	    return rmse;
	}
	//  * the estimation vector size should equal ground truth vector size
	//accumulate squared residuals
	for(int i=0; i < estimations.size(); ++i){
        // ... your code here
        VectorXd difference = estimations[i] - ground_truth[i];
        difference = difference.array() * difference.array();
        rmse = rmse + difference;
	}

	//calculate the mean
	rmse /= estimations.size();


	//calculate the squared root
	rmse = rmse.array().sqrt();

	//return the result
	return rmse;

}

MatrixXd Tools::MeasurementJacobianMatrix(const VectorXd& x_state) {
  MatrixXd Hj(3,4);
	//recover state parameters
	float px = x_state(0);
	float py = x_state(1);
	float vx = x_state(2);
	float vy = x_state(3);

	//pre-compute a set of terms to avoid repeated calculation
	float c1 = px*px+py*py;
	float c2 = sqrt(c1);
	float c3 = (c1*c2);

	//check division by zero
	if((fabs(c1) < .0001) || (fabs(c2) < .0001) || (fabs(c3) < .0001)) {
		Hj << 0,0,0,0,
		      0,0,0,0,
		      0,0,0,0;
		return Hj;
	}

	//compute the Jacobian matrix
	Hj << (px/c2), (py/c2), 0, 0,
		  -(py/c1), (px/c1), 0, 0,
		  py*(vx*py - vy*px)/c3, px*(px*vy - py*vx)/c3, px/c2, py/c2;

	return Hj;
}

MatrixXd Tools::ProcessCovarianceMatrix(float dt, float noise_ax, float noise_ay) {
	float dt2 = dt*dt;
	float dt3 = dt2*dt;
	float dt4 = dt3*dt;

	MatrixXd Q = MatrixXd(4,4);

	Q << dt4/4*noise_ax, 0, dt3/2*noise_ax, 0,
	     0, dt4/4*noise_ay, 0, dt3/2*noise_ay,
	     dt3/2*noise_ax, 0, dt2*noise_ax, 0,
	     0, dt3/2*noise_ay, 0, dt2*noise_ay;

	return Q;
}

// Makes the transition prediction matrix with the specified time difference
MatrixXd Tools::TransitionMatrix(float time_difference) {

  MatrixXd F = MatrixXd(4,4);
  F << 1, 0, time_difference, 0,
       0, 1, 0, time_difference,
       0, 0, 1, 0,
       0, 0, 0, 1;

  return F;
}

// Turns a 3x1 radial vector into a 4x1 cartesian vector
Eigen::VectorXd Tools::RadialToCartesian(const VectorXd &z) {
  // Make sure we have a 3x1 vector
  assert(z.size() == 3);

  float range = z[0], bearing = z[1], radial_velocity = z[2];
  float sin_bearing = sin(bearing);
  float cos_bearing = cos(bearing);

  float px = cos_bearing * range;
  float py = sin_bearing * range;
  float vx = cos_bearing * radial_velocity;
  float vy = sin_bearing * radial_velocity;

  VectorXd x = VectorXd(4);
  x << px, py, vx, vy;
  return x;

}

Eigen::VectorXd Tools::CartesianToRadial(const Eigen::VectorXd &x) {
	VectorXd h = VectorXd(3);
    float px = x[0], py = x[1], vx = x[2], vy = x[3];
    float range = pow(px * px + py * py, .5);
    float bearing;
    float radial_velocity;

    // Check for zero division error
    if (px == 0 && py == 0)
    	bearing = 0;
    else
    	bearing = atan2(py, px);

    if (range == 0)
    	radial_velocity = 0;
    else
    	radial_velocity = (px*vx + py*vy)/range;

    h << range, bearing, radial_velocity;
    return h;
}
