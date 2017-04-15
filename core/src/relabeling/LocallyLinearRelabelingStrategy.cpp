/*-------------------------------------------------------------------------------
 This file is part of gradient-forest...
 #-------------------------------------------------------------------------------*/

#include "utility.h"
#include "LocallyLinearRelabelingStrategy.h"

LocallyLinearRelabelingStrategy::LocallyLinearRelabelingStrategy():
    lambda(0.01) {}

LocallyLinearRelabelingStrategy::LocallyLinearRelabelingStrategy(double lambda):
    lambda(lambda) {}


std::unordered_map<size_t, double> LocallyLinearRelabelingStrategy::relabel_outcomes(
                                                                                     const Observations& observations,
                                                                                     const std::vector<size_t>& node_sampleIDs) {
    
    size_t num_samples = node_sampleIDs.size(); // is there a reason Julie didn't do this earlier to get num_samples?
    
    // find number of covariates: is there a cleaner way?? ********
    Eigen::RowVectorXf observation;
    observation << myMap.begin()->second;
    const size_t num_covariates = observation.size(); // assuming we already have the vector of 1s, this includes the constant
    
    Eigen::Matrix<double, num_covariates, 1> theta;
    
    // extract matrix X  and vector Y
    Eigen::Matrix<double, num_samples, num_covariates> X = Eigen::Matrix<double, num_samples, num_covariates>::Zero();
    Eigen::Matrix<double, num_samples, 1> Y = Eigen::Matrix<double, num_samples, 1>::Zero();
    
    int i = 0;
    for (size_t sampleID : node_sampleIDs) {
        Eigen::RowVectorXf row = observations.get(Observations::COVARIATES, sampleID);
        X.block<1,num_covariates>(i, 0) << row;
        Y(i) << observations.get(Observations::OUTCOME, sampleID);
        ++i;
    }
    
    // Identity matrix
    Eigen::Matrix<double, num_covariates, num_covariates> Id = Eigen::Matrix<double, num_covariates, num_covariates>::Identity();
    
    Eigen::Matrix<double, num_covariates, num_covariates> J = Id;
    J(0,0) = 0;
    
    // Pre-compute ridged variance estimate and its inverse
    Eigen::Matrix<double, num_covariates, num_covariates> M = X.transpose()*X + J*lambda;
    Eigen::Matrix<double, num_covariates, num_covariates> M_inverse = M.colPivHouseholderQr().solve(Id);
    
    theta = M*X.transpose()*Y;
    
    double rho;
    std::unordered_map<size_t, double> relabeled_observations;
    for (size_t sampleID : node_sampleIDs) {
        double response = observations.get(Observations::OUTCOME, sampleID);
        std::vector<double> xi = observations.get(Observations::COVARIATES, sampleID);
        
        rho << M_inverse*(Y-theta*xi);
        relabeled_observations[sampleID] = rho;
    }
    return relabeled_observations;
}

