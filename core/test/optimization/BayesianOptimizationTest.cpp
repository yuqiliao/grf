// please see the explanation in the documentation
// http://www.resibots.eu/limbo

#define USE_NLOPT
#include <iostream>
#include <limbo/limbo.hpp>

#include "commons/utility.h"
#include "forest/ForestPredictor.h"
#include "forest/ForestPredictors.h"
#include "forest/ForestTrainer.h"
#include "forest/ForestTrainers.h"
#include "utilities/ForestTestUtilities.h"

#include "catch.hpp"


struct Params {
  struct bayes_opt_boptimizer : public limbo::defaults::bayes_opt_boptimizer {};

  struct opt_gridsearch: public limbo::defaults::opt_gridsearch {};
  struct opt_nloptnograd : public limbo::defaults::opt_nloptnograd {
    BO_PARAM(double, fun_tolerance, 1e-6);
    BO_PARAM(double, xrel_tolerance, 1e-6);
  };

  struct kernel : public limbo::defaults::kernel {
    BO_PARAM(double, noise, 0.01);
    BO_PARAM(double, optimize_noise, true);
  };

  struct bayes_opt_bobase : public limbo::defaults::bayes_opt_bobase {};
  struct kernel_maternfivehalves : public limbo::defaults::kernel_maternfivehalves {};
  struct init_randomsampling : public limbo::defaults::init_randomsampling {};

  struct stop_maxiterations {
    BO_PARAM(int, iterations, 500);
  };

  struct acqui_ucb : public limbo::defaults::acqui_ucb {};
};

struct ObjectiveFunction {
  BO_PARAM(size_t, dim_in, 4);
  BO_PARAM(size_t, dim_out, 1);

  Data* data;
  size_t outcome_index;
  ObjectiveFunction(Data* data, size_t outcome_index) :
      data(data),
      outcome_index(outcome_index) {}

  Eigen::VectorXd operator()(const Eigen::VectorXd &x) const {
    double alpha = x[0] / 2;
    uint ci_group_size = 1;
    uint min_node_size = (uint) (x[1] * 100);
    uint mtry = (uint) (x[2] * data->get_num_cols());
    double sample_fraction = 0.20 + 0.8 * x[3];

    // Train a regression forest, and make OOB predictions.
    ForestTrainer trainer = ForestTrainers::regression_trainer(data, outcome_index, alpha);
    ForestTestUtilities::init_trainer(trainer, true, ci_group_size,
                                      min_node_size, mtry, sample_fraction);

    Forest forest = trainer.train(data);
    ForestPredictor predictor = ForestPredictors::regression_predictor(4, 1);
    std::vector<Prediction> predictions = predictor.predict_oob(forest, data);

    // Calculate and return the mean squared error.
    double difference = 0;
    for (size_t i = 0; i < predictions.size(); ++i) {
      double actual_outcome = data->get(i, outcome_index);

      double tree_mean = 0.0;
      for (std::shared_ptr<Tree> tree : forest.get_trees()) {


      }
      double predicted_outcome = predictions[i].get_predictions().at(0);
      difference += (actual_outcome - predicted_outcome) * (actual_outcome - predicted_outcome);
    }

    double mean_squared_error = difference / predictions.size();
    return limbo::tools::make_vector(1 - mean_squared_error);
  }
};

TEST_CASE("bayes optimization completes without error", "[optimization]") {
    // we use the default acquisition function / model / stat / etc.

  Data* data = load_data("test/forest/resources/performance_test.csv");
  size_t num_cols = data->get_num_cols();
  size_t outcome_index = 10;

//  typedef limbo::kernel::MaternFiveHalves<Params> Kernel_t;
//  typedef limbo::mean::Data<Params> Mean_t;
//  typedef limbo::model::GP<Params, Kernel_t, Mean_t> GP_t;
//  typedef limbo::acqui::UCB<Params, GP_t> Acqui_t;
//
//  limbo::bayes_opt::BOptimizer<Params, limbo::modelfun<GP_t>, limbo::acquifun<Acqui_t>> optimizer;

  limbo::bayes_opt::BOptimizer<Params> optimizer;

  ObjectiveFunction objective(data, outcome_index);
  optimizer.optimize(objective);

  std::cout << "Best alpha: " << optimizer.best_sample()[0] / 2 << std::endl
            << " Best min_node_size: " << (uint) (optimizer.best_sample()[1] * 100) << std::endl
            << " Best mtry: " << (uint) (optimizer.best_sample()[2] * num_cols) << std::endl
            << " Best sample_fraction: " << 0.20 + 0.80 * optimizer.best_sample()[3] << std::endl
            << " Best observation: " << optimizer.best_observation()(0) << std::endl;
}
