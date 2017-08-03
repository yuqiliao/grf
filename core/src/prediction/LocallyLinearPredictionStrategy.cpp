
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


LocallyLinearPredictionStrategy::LocallyLinearPredictionStrategy(const Data *data, const Data *test_data, std::vector<double> lambda):
    data(data),
    test_data(test_data),
    lambda(lambda){
};

const size_t LocallyLinearPredictionStrategy::OUTCOME = 0;

size_t LocallyLinearPredictionStrategy::prediction_length() {
    return 1;
}

Prediction LocallyLinearPredictionStrategy::predict(size_t sampleID,
                                                    const std::vector<double>& average_prediction_values,
                                                    const std::unordered_map<size_t, double>& weights_by_sampleID,
                                                    const Observations& observations) {
    size_t n;
    size_t p;
    
    n = observations.get_num_samples();
    p = test_data->get_num_cols();
    
    Eigen::MatrixXf weights(n,n);
    weights = Eigen::MatrixXf::Zero(n,n);
    
    for (auto it = weights_by_sampleID.begin(); it != weights_by_sampleID.end(); ++it){
        size_t i = it->first;
        double weight = it->second;
        weights(i,i) = weight;
    }
    
    // create test point vector
    Eigen::MatrixXf test_point(1, p);
    for(size_t j=0; j<p; ++j){
        test_point(j) = test_data->get(sampleID,j);
    }
    
    // generate design matrix X and responses Y as Eigen objects
    Eigen::MatrixXf X(n, p+1);
    Eigen::MatrixXf Y(n, 1);
    
    for (size_t i=0; i<n; ++i) {
        for(size_t j=0; j<p; ++j){
            X(i,j+1) = test_point(j) - data->get(i,j);
        }
        Y(i) = observations.get(Observations::OUTCOME, i);
        X(i, 0) = 1;
    }
    
    Eigen::MatrixXf Id(p+1,p+1);
    Eigen::MatrixXf J(p+1,p+1);
    
    Id = Eigen::MatrixXf::Identity(p+1,p+1);
    J = Eigen::MatrixXf::Identity(p+1,p+1);
    J(0,0) = 0;
    
    // Insert ridge correction
    for(size_t i = 1; i<p+1; ++i){
        J(i,i) *= lambda[i-1];
    }
    
    // Pre-compute ridged variance estimate and its inverse
    Eigen::MatrixXf M(p+1,p+1);
    Eigen::MatrixXf M_inverse(p+1,p+1);
    M  = X.transpose()*weights*X + J;
    M_inverse = M.colPivHouseholderQr().solve(Id);
    
    // create theta vector
    Eigen::MatrixXf theta(p+1,1);
    theta = M_inverse*X.transpose()*weights*Y;
    
    std::vector<double> yhat_vector;
    yhat_vector.push_back(theta(0));
    
    return Prediction(yhat_vector);
}

Prediction LocallyLinearPredictionStrategy::predict_with_variance(size_t sampleID,
                                                                  const std::vector<std::vector<size_t>>& leaf_sampleIDs,
                                                                  const Observations& observations,
                                                                  uint ci_group_size) {
    throw std::runtime_error("Variance estimates are not yet implemented.");
}

bool LocallyLinearPredictionStrategy::requires_leaf_sampleIDs(){
    return true;
}

PredictionValues LocallyLinearPredictionStrategy::precompute_prediction_values(
                                                                const std::vector<std::vector<size_t>>& leaf_sampleIDs,
                                                                const Observations& observations){
    return PredictionValues();
}
