
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


LocallyLinearPredictionStrategy::LocallyLinearPredictionStrategy(const Data *data, const Data *test_data, double lambda):
    //totally not sure that this is right
    data(data),
    test_data(test_data),
    lambda(lambda){
};

const size_t LocallyLinearPredictionStrategy::OUTCOME = 0;

size_t LocallyLinearPredictionStrategy::prediction_length() {
    return 1;
    //return data->get_num_rows(); // this should change. to 1? to test data something? check other methods.
}

Prediction LocallyLinearPredictionStrategy::predict(size_t sampleID,
                                                    const std::vector<double>& average_prediction_values,
                                                    const std::unordered_map<size_t, double>& weights_by_sampleID,
                                                    const Observations& observations) {
    size_t n;
    n = observations.get_num_samples(); // observations == ys
    
    Eigen::MatrixXf weights(n,1);
    weights = Eigen::MatrixXf::Zero(n,1);
    
    for (auto it = weights_by_sampleID.begin(); it != weights_by_sampleID.end(); ++it){
        size_t i = it->first;
        double weight = it->second;
        weights(i) = weight;
    }
    
    /*
    std::vector<double> weights_vector;
    for(size_t i=0; i<n; ++i){
        weights_vector.push_back(weights(i));
    }
     this may not be necessary, I might want it as an Eigen object actually.
     */
    
    size_t p = data->get_num_cols();
    
    // generate training data as Eigen objects
    
    Eigen::MatrixXf X(n, p);
    Eigen::MatrixXf Y(n, 1);
    
    for (size_t i=0; i<n; ++i) {
        for(size_t j=0; j<p; ++j){
            X(i,j) = data->get(i,j);
        }
        Y(i) = observations.get(Observations::OUTCOME, sampleID);
        ++i;
    }
    
    Eigen::MatrixXf Id(p,p);
    Eigen::MatrixXf J(p,p);
    
    Id = Eigen::MatrixXf::Identity(p,p);
    J = Eigen::MatrixXf::Identity(p,p);
    J(0,0) = 0;
    
    
    // Pre-compute ridged variance estimate and its inverse
    Eigen::MatrixXf M(p,p);
    Eigen::MatrixXf M_inverse(p,p);
    M = X.transpose()*X + J*lambda;
    M_inverse = M.colPivHouseholderQr().solve(Id);
    
    // create theta vector
    Eigen::MatrixXf theta(p,1);
    theta = M_inverse*X.transpose()*Y;
    
    //
    Eigen::MatrixXf test_point(1, p);
    for(size_t j=0; j<p; ++j){
        test_point(j) = test_data->get(sampleID,j);
    }
    Eigen::MatrixXf yhat;
    yhat = test_point.transpose()*theta;
    
    std::vector<double> yhat_vector;
    for(size_t i=0; i<p; ++i){
        yhat_vector[i] = yhat(i);
    }
    
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
