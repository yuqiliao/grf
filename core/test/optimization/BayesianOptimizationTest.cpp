// please see the explanation in the documentation
// http://www.resibots.eu/limbo

#include <iostream>
#include <limbo/bayes_opt/boptimizer.hpp>

#include "commons/utility.h"
#include "forest/ForestPredictor.h"
#include "forest/ForestPredictors.h"
#include "forest/ForestTrainer.h"
#include "forest/ForestTrainers.h"
#include "utilities/ForestTestUtilities.h"

#include "catch.hpp"


struct Params {
    struct bayes_opt_boptimizer : public limbo::defaults::bayes_opt_boptimizer {};

    struct opt_gridsearch : public limbo::defaults::opt_gridsearch {};

    struct kernel : public limbo::defaults::kernel {
        BO_PARAM(double, noise, 0.001);
    };

    struct bayes_opt_bobase : public limbo::defaults::bayes_opt_bobase {};

    struct kernel_maternfivehalves : public limbo::defaults::kernel_maternfivehalves {};

    struct init_randomsampling : public limbo::defaults::init_randomsampling {};

    struct stop_maxiterations : public limbo::defaults::stop_maxiterations {};

    struct acqui_ucb : public limbo::defaults::acqui_ucb {};
};

struct ObjectiveFunction {
  // number of input dimension (x.size())
  BO_PARAM(size_t, dim_in, 1);
  // number of dimensions of the result (res.size())
  BO_PARAM(size_t, dim_out, 1);

  Data* data;
  ObjectiveFunction(Data* data) : data(data) {}

  // the function to be optimized
  Eigen::VectorXd operator()(const Eigen::VectorXd &x) const {

    // Train a regression forest, and make OOB predictions.
    uint outcome_index = 10;
    double alpha = 0.10;

    ForestTrainer trainer = ForestTrainers::regression_trainer(data, outcome_index, alpha);
    ForestTestUtilities::init_honest_trainer(trainer);

    Forest forest = trainer.train(data);
    ForestPredictor predictor = ForestPredictors::regression_predictor(4, 1);
    std::vector<Prediction> predictions = predictor.predict_oob(forest, data);

    // Calculate and return the mean squared error.
    double difference = 0;
    for (int i = 0; i < predictions.size(); ++i) {
      double y_real = data->get(i, outcome_index);
      double y_pred = predictions[i].get_predictions().at(0);
      //std::cout << "*" << i << y_real-y_pred << std::endl;
      difference += (y_real - y_pred) * (y_real - y_pred);
    }

    double mean_squared_error = difference / predictions.size();
    return limbo::tools::make_vector(mean_squared_error);
  }
};

TEST_CASE("bayes optimization completes without error", "[optimization]") {
    // we use the default acquisition function / model / stat / etc.

  Data* data = load_data("test/forest/resources/gaussian_data.csv");
  limbo::bayes_opt::BOptimizer<Params> optimizer;

  ObjectiveFunction objective(data);
  optimizer.optimize(objective);

  std::cout << "Best sample: " << optimizer.best_sample()(0) << " - Best observation: "
            << optimizer.best_observation()(0) << std::endl;
}
