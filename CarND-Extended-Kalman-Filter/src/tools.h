#ifndef TOOLS_H_
#define TOOLS_H_
#include <vector>
#include "Eigen/Dense"

class Tools {
public:
  /**
  * Constructor.
  */
  Tools();

  /**
  * Destructor.
  */
  virtual ~Tools();

  /**
  * A helper method to calculate RMSE.
  */
  static Eigen::VectorXd CalculateRMSE(const std::vector<Eigen::VectorXd> &estimations, const std::vector<Eigen::VectorXd> &ground_truth);

  /**
  * A helper method to calculate Jacobians.
  */
  static Eigen::MatrixXd MeasurementJacobianMatrix(const Eigen::VectorXd& x_state);

  /**
  * Calculates the Q matrix from the diven timestamp and noise values)
  **/
  static Eigen::MatrixXd ProcessCovarianceMatrix(float dt, float noise_ax, float noise_ay);

  static Eigen::MatrixXd TransitionMatrix(float time_difference);

  static Eigen::VectorXd RadialToCartesian(const Eigen::VectorXd &z);

  static Eigen::VectorXd CartesianToRadial(const Eigen::VectorXd &x);

};

#endif /* TOOLS_H_ */
