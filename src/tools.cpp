#include "tools.h"
#include <iostream>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
   VectorXd rmse(4);
   rmse<<0,0,0,0;
   if(estimations.size() == 0 || estimations.size() != ground_truth.size()) {
      cout<<"estimations are either zero sized or not equal to ground truth array.";
      return rmse;
   }
   for(int i=0; i<estimations.size(); i++) {
      VectorXd residuals = estimations[i] - ground_truth[i];
      residuals = residuals.array() * residuals.array();
      rmse += residuals;
   }

   rmse /= rmse/estimations.size();
   rmse = rmse.array().sqrt();
   return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  Matrix Hj(3,4);
  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);

  float c1 = px*px + py*py;
  float c1_root = sqrt(c1);
  float c1_32 = c1*c1_root;

  if(c1 < 0.0001) {
     cout<<"Jacobian: dividing by zero. Returning empty matrix";
     return Hj;
  }
  Hj << (px/c1_root), (py/c1_root), 0, 0,
      -(py/c1), (px/c1), 0, 0,
      py*(vx*py - vy*px)/c1_32, px*(px*vy - py*vx)/c1_32, px/c1_root, py/c1_root;

  return Hj;
}
