/*-------------------------------------------------------------------------------
 This file is part of gradient-forest...
 #-------------------------------------------------------------------------------*/

#include "utility.h"
#include "eigen3/Eigen/Dense"
#include "LocallyLinearRelabelingStrategy.h"

LocallyLinearRelabelingStrategy::LocallyLinearRelabelingStrategy(double lambda,
                                                                 const Data *data):
    lambda(lambda),
    data(data){}

// not sure about the syntax above here; must check

std::unordered_map<size_t, double> LocallyLinearRelabelingStrategy::relabel_outcomes(const Observations& observations,
                                                                                     const std::vector<size_t>& node_sampleIDs) {
    
    // find number of samples and covariates
    size_t num_samples = node_sampleIDs.size();
    size_t p = data->get_num_cols(); // check on usage
   
    // initialize theta
    Eigen::MatrixXf X(num_samples, p);
    Eigen::MatrixXf Y(num_samples,1);
    
    int i = 0;
    for (size_t sampleID : node_sampleIDs) {
        for(size_t j; j<p; ++j){
            X(i,j) = data->get(i,j);
        }
        // Eigen::RowVectorXf row = observations.get(Observations::COVARIATES, sampleID);
        // X.block<1,p>(i, 0) << row;
        Y(i) = observations.get(Observations::OUTCOME, sampleID);
        ++i;
    }
    
    // Identity matrix
    Eigen::MatrixXf Id(p,p);
    Eigen::MatrixXf J(p,p);
    
    Id = Eigen::MatrixXf::Identity(p,p);
    J = Eigen::MatrixXf::Identity(p,p);
    J(0,0) = 0;
    
    
    Eigen::MatrixXf theta(p,1);
    
    // Pre-compute ridged variance estimate and its inverse
    Eigen::MatrixXf M(p,p);
    Eigen::MatrixXf M_inverse(p,p);
    M = X.transpose()*X + J*lambda;
    M_inverse = M.colPivHouseholderQr().solve(Id);
    
    theta = M_inverse*X.transpose()*Y;
    
    std::unordered_map<size_t, double> relabeled_observations;
    for (size_t sampleID : node_sampleIDs) {
        double response = observations.get(Observations::OUTCOME, sampleID);
        Eigen::MatrixXf xi(1,p);
        xi = X.block(1,p,sampleID,0);
    
        Eigen::MatrixXf temp(1,1);
        temp = xi*theta;
        
        double difference;
        difference = response-temp(1);
        
        Eigen::MatrixXf rho(p,1);
        rho = M_inverse*xi.transpose()*difference;
        relabeled_observations[sampleID] = rho(1);
    }
    return relabeled_observations;
}

