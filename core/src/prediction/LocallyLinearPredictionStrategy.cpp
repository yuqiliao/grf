
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

Prediction LocallyLinearPredictionStrategy::predict(size_t sampleID, // is this the test point?
                                                    const std::vector<std::vector<size_t>> &leaf_sampleIDs, // note: vector of vectors
                                                    const Observations& observations) {
    
    size_t num_leaves = leaf_sampleIDs.size() // number of trees, or number of leaves containing x0?
    size_t n = observations.size() // Waiting to hear from Julie that this method exists here
    
    // initialize weight matrix of 0's; we'll update the diagonal
    Eigen::Matrix<float, n, n> weights = Eigen::Matrix<float, n, n>::Zero();
    
    for(size_t i=0; i<leaf_sampleIDs.size(); ++i){
        size_t leaf_size = leaf_sampleIDs[i].size(); //size of current leaf
        size_t denominator = leaf_size*num_leaves;
        if(leaf_size == 0){
            continue;
        }
        for(auto& sampleID : leaf_sampleIDs[i]){
            // are sampleIDs in 1...n ? ALL of this code assumes yes so must double check.
            weights(sampleID, sampleID) += 1 / denominator;
        }
    }
    
    // we now have a complete diagonal weight matrix weights
    // we now move on to the local linear prediction
    // currently assuming X has been centered already
    
    size_t p = observations.get(Observations::COVARIATES,1).size() // double check this method
    Eigen::Matrix<float, n, p> X;
    Eigen::Matrix<float, n, 1> Y;
    
    // loop through observations to fill in p, X, Y
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
