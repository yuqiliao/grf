
/*-------------------------------------------------------------------------------
 This file is part of gradient-forest.
 gradient-forest is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.
 gradient-forest is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 GNU General Public License for more details.
 You should have received a copy of the GNU General Public License
 along with gradient-forest. If not, see <http://www.gnu.org/licenses/>.
 #-------------------------------------------------------------------------------*/



#include <cmath>
#include <string>
#include "prediction/LocallyLinearPredictionStrategy.h"

const size_t LocallyLinearPredictionStrategy::OUTCOME = 0;

size_t LocallyLinearPredictionStrategy::prediction_length() {
    return 1;
}

LocallyLinearRelabelingStrategy::LocallyLinearPredictionStrategy():
lambda(0.01) {}

LocallyLinearRelabelingStrategy::LocallyLinearPredictionStrategy(double lambda):
lambda(lambda) {}

Prediction LocallyLinearPredictionStrategy::predict(size_t sampleID,
                                                    const std::unordered_map<size_t, double>& weights_by_sampleID,
                                                    const Observations& observations) {
    
    size_t n = observations.get_num_samples() // usage correct?
    
    // initialize weight matrix
    Eigen::Matrix<float, n, n> weights = Eigen::Matrix<float, n, n>::Zero();
    
    
    // loop through all leaves and update weight matrix
    for (auto it = weights_by_sampleID.begin(); it != weights_by_sampleID.end(); ++it){
        size_t i = it->first;
        float weight = it->second;
        weights(i,i) = weight;
    }
    
    // we now move on to the local linear prediction assuming X has been formatted correctly
    
    size_t p = observations.get(Observations::COVARIATES,1).size() // double check this method
    Eigen::Matrix<float, n, p> X;
    Eigen::Matrix<float, n, 1> Y;
    
    // loop through observations to fill in X, Y
    for(size_t i=0; i<n; ++i){
        Eigen::Matrix<float, 1, p> temp_row = observations.get(Observations::COVARIATES, i);
        X.block<1,p>(i,0) = temp_row;
        Y[i] = observations.get(Observations::OUTCOME, i);
    }
    
    // Pre-compute M = X^T X + lambda J
    Eigen::Matrix<float, n, n> Id = Eigen::Matrix<float, n, n>::Identity();
    
    Eigen::Matrix<float, n, n> J = Id;
    J(0,0) = 0;
    
    Eigen::Matrix<float, n, n> M = X.transpose()*X + J*lambda;
    Eigen::Matrix<float, n, n> M.inverse = M.colPivHouseholderQr().solve(Id);
    
    theta = M*X.transpose()*Y;
    Eigen::Matrix<float, n, 1> Y_hat = test_point.transpose()*theta;
    double predictions = Y_hat(1) // need to be Y_hat(1,1)? 
    
    return Prediction(predictions);
}
