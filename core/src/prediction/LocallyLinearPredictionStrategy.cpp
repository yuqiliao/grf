
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


LocallyLinearPredictionStrategy::LocallyLinearPredictionStrategy(const Data *data):
    data(data){
};

const size_t LocallyLinearPredictionStrategy::OUTCOME = 0;

size_t LocallyLinearPredictionStrategy::prediction_length() {
    return data->get_num_rows();
}

Prediction LocallyLinearPredictionStrategy::predict(size_t sampleID,
                                                    const std::vector<double>& average_prediction_values,
                                                    const std::unordered_map<size_t, double>& weights_by_sampleID,
                                                    const Observations& observations) {
    size_t n;
    n = observations.get_num_samples();

    Eigen::MatrixXf weights(n,1);
    weights = Eigen::MatrixXf::Zero(n,1);
    
    for (auto it = weights_by_sampleID.begin(); it != weights_by_sampleID.end(); ++it){
        size_t i = it->first;
        double weight = it->second;
        weights(i) = weight;
    }
    
    std::vector<double> weights_vector;
    for(size_t i=0; i<n; ++i){
        weights_vector.push_back(weights(i));
    }
    
    return Prediction(weights_vector);
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
