
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
#include <vector>
#include <iostream>
#include "eigen3/Eigen/Dense"
#include "commons/utility.h"
#include "commons/Observations.h"
#include "prediction/LocallyLinearPredictionStrategy.h"


LocallyLinearPredictionStrategy::LocallyLinearPredictionStrategy(double lambda, const Data *data):
    lambda(lambda),
    data(data){
};

/*
 // would be nice to have optional lambda but can force all this in R so not necessary right now 
 LocallyLinearPredictionStrategy::LocallyLinearPredictionStrategy(const Data *data):
    lambda(0.1),
    Data(data){
};
 */

const size_t LocallyLinearPredictionStrategy::OUTCOME = 0;

size_t LocallyLinearPredictionStrategy::prediction_length() {
    return 1;
}

Prediction LocallyLinearPredictionStrategy::predict(size_t sampleID,
                                                    const std::vector<double>& average_prediction_values,
                                                    const std::unordered_map<size_t, double>& weights_by_sampleID,
                                                    const Observations& observations) {
    
    //Data input_data;
    //input_data = *data;
    
    size_t n;
    size_t p;
    n = observations.get_num_samples(); // usage correct?
    p = data->get_num_cols(); // use correct?
    
    // initialize weight matrix
    Eigen::MatrixXf weights(n,n);
    
    // loop through all leaves and update weight matrix
    for (auto it = weights_by_sampleID.begin(); it != weights_by_sampleID.end(); ++it){
        size_t i = it->first;
        float weight = it->second;
        weights(i,i) = weight;
    }
    
    // we now move on to the local linear prediction assuming X has been formatted correctly
    
    //size_t p = Data::num_cols(data); // double check this method
    Eigen::MatrixXf X(n,p);
    Eigen::MatrixXf Y(n,1);
    
    // loop through observations to fill in X, Y, weights
    for(size_t i=0; i<n; ++i){
        for(size_t j=0; j<p; ++j){
            //X.block<1,1>(i,j) << data->get(i,j);
            //X.block<1,1>(i,j) = input_data.get(i,j);
            X(i,j) = data->get(i,j);
            if(i != j){
                weights(i,j) = 0;
            }
        }
        Y.block<1,1>(i,0) << observations.get(Observations::OUTCOME, i);
    }
    
    // Pre-compute M = X^T X + lambda J
    //float lambda = 0.01;
    Eigen::MatrixXf J(p,p);
    Eigen::MatrixXf Id(p,p);
    J = Eigen::MatrixXf::Identity(p,p);
    Id = Eigen::MatrixXf::Identity(p,p);
    J(0,0) = 0;
    
    Eigen::MatrixXf M(p,p);
    M = X.transpose()*weights*X + J*lambda;
    Eigen::MatrixXf M_inverse(p,p);
    M_inverse = M.colPivHouseholderQr().solve(Id);
    
    Eigen::MatrixXf theta(1,p);
    theta = M_inverse*X.transpose()*weights*Y;
    
    std::vector<double> theta_vector;
    for(size_t i=1; i<p; ++i){
        theta_vector[i] = theta(i);
    }
    
    return Prediction(theta_vector); // do not have test point yet; returning theta for now instead
}

// now defining dummy methods to see if the compiler stops complaining to me about pure virtual methods

Prediction LocallyLinearPredictionStrategy::predict_with_variance(size_t sampleID,
                                                                  const std::vector<std::vector<size_t>>& leaf_sampleIDs,
                                                                  const Observations& observations) {
    throw std::runtime_error("Variance estimates are not yet implemented.");
}

bool LocallyLinearPredictionStrategy::requires_leaf_sampleIDs(){
    return false;
}

PredictionValues precompute_prediction_values(const std::vector<std::vector<size_t>> leaf_sampleIDs,
                                              const Observations& observations){
    throw std::runtime_error("Not implemented yet.");
}
