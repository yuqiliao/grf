#include <Rcpp.h>
#include <vector>
#include <sstream>
#include <map>

#include "RcppUtilities.h"
#include "commons/globals.h"
#include "forest/ForestTrainers.h"
#include "forest/ForestPredictors.h"

// NOW LOCALLY LINEAR FOREST BINDINGS

// [[Rcpp::export]]
Rcpp::List locally_linear_train(Rcpp::NumericMatrix input_data,
                            size_t outcome_index,
                            Rcpp::RawMatrix sparse_data,
                            std::vector <std::string> variable_names,
                            std::vector<double> lambda,
                            unsigned int mtry,
                            unsigned int num_trees,
                            bool verbose,
                            unsigned int num_threads,
                            unsigned int min_node_size,
                            bool sample_with_replacement,
                            bool keep_inbag,
                            double sample_fraction,
                            std::vector<size_t> no_split_variables,
                            unsigned int seed,
                            bool honesty,
                            unsigned int ci_group_size) {
 
  Data* data = RcppUtilities::convert_data(input_data, sparse_data, variable_names);
    
  Data* test_data = data; //hacky?
    
  ForestTrainer trainer = ForestTrainers::locally_linear_trainer(data, test_data, lambda, outcome_index);
  RcppUtilities::initialize_trainer(trainer, mtry, num_trees, num_threads, min_node_size,
      sample_with_replacement, sample_fraction, no_split_variables, seed, honesty, ci_group_size);
    
  Forest forest = trainer.train(data);

  Rcpp::List result;
  Rcpp::RawVector serialized_forest = RcppUtilities::serialize_forest(forest);
  result.push_back(serialized_forest, RcppUtilities::SERIALIZED_FOREST_KEY);
  result.push_back(forest.get_trees().size(), "num.trees");

  delete data;
 
  return result;
}

// [[Rcpp::export]]
Rcpp::NumericMatrix locally_linear_predict(Rcpp::List forest,
                                       Rcpp::NumericMatrix input_data,
                                       Rcpp::RawMatrix sparse_data,
                                       Rcpp::NumericMatrix train,
                                       Rcpp::RawMatrix sparse_training_data,
                                       std::vector<double> lambda,
                                       std::vector<std::string> variable_names,
                                       unsigned int num_threads) {
  Data *test_data = RcppUtilities::convert_data(input_data, sparse_data, variable_names);
  Data *data = RcppUtilities::convert_data(train, sparse_training_data, variable_names);
    
  Forest deserialized_forest = RcppUtilities::deserialize_forest(
      forest[RcppUtilities::SERIALIZED_FOREST_KEY]);
    
  ForestPredictor predictor = ForestPredictors::locally_linear_predictor(num_threads, data, test_data, lambda);
  std::vector<Prediction> predictions = predictor.predict(deserialized_forest, test_data);
  Rcpp::NumericMatrix result = RcppUtilities::create_prediction_matrix(predictions);

  delete data;
  delete test_data;
  return result;
}

// [[Rcpp::export]]
Rcpp::NumericMatrix locally_linear_predict_oob(Rcpp::List forest,
                                           Rcpp::NumericMatrix input_data,
                                           Rcpp::RawMatrix sparse_data,
                                           Rcpp::NumericMatrix train,
                                           Rcpp::RawMatrix sparse_training_data,
                                           std::vector<double> lambda,
                                           std::vector<std::string> variable_names,
                                           unsigned int num_threads) {
  Data *test_data = RcppUtilities::convert_data(input_data, sparse_data, variable_names);
  Data *data = RcppUtilities::convert_data(train, sparse_training_data, variable_names);
    
  Forest deserialized_forest = RcppUtilities::deserialize_forest(
      forest[RcppUtilities::SERIALIZED_FOREST_KEY]);

  ForestPredictor predictor = ForestPredictors::locally_linear_predictor(num_threads, data, test_data, lambda);
  std::vector<Prediction> predictions = predictor.predict_oob(deserialized_forest, test_data);
  Rcpp::NumericMatrix result = RcppUtilities::create_prediction_matrix(predictions);


  delete data;
  delete test_data;
  return result;
}
