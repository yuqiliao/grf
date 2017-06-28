#' Regression forest
#' 
#' Trains a regression forest that can be used to estimate
#' the conditional mean function mu(x) = E[Y | X = x]
#'
#' @param X The covariates used in the regression.
#' @param Y The outcome.
#' @param sample.fraction Fraction of the data used to build each tree.
#'                        Note: If honesty is used, these subsamples will
#'                        further be cut in half.
#' @param mtry Number of variables tried for each split.
#' @param num.trees Number of trees grown in the forest. Note: Getting accurate
#'                  confidence intervals generally requires more trees than
#'                  getting accurate predictions.
#' @param num.threads Number of threads used in training. If set to NULL, the software
#'                    automatically selects an appropriate amount.
#' @param min.node.size Minimum number of observations in each tree leaf.
#' @param honesty Should honest splitting (i.e., sub-sample splitting) be used?
#' @param ci.group.size The forst will grow ci.group.size trees on each subsample.
#'                      In order to provide confidence intervals, ci.group.size must
#'                      be at least 2.
#' @param alpha Maximum imbalance of a split.
#' @param seed The seed of the c++ random number generator.
#'
#' @return A trained regression forest object.
#' @export
regression_forest <- function(X, Y, sample.fraction = 0.5, mtry = ceiling(2*ncol(X)/3), 
                              num.trees = 2000, num.threads = NULL, min.node.size = NULL,
                              honesty = TRUE, ci.group.size = 2, alpha = 0.05, seed = NULL) {
    
    validate.X(X)
    if(length(Y) != nrow(X)) { stop("Y has incorrect length.") }
    
    mtry <- validate.mtry(mtry)
    num.threads <- validate.num.threads(num.threads)
    min.node.size <- validate.min.node.size(min.node.size)
    sample.fraction <- validate.sample.fraction(sample.fraction)
    seed <- validate.seed(seed)
    
    sparse.data <- as.matrix(0)
    no.split.variables <- numeric(0)
    sample.with.replacement <- FALSE
    verbose <- FALSE
    keep.inbag <- FALSE
    
    input.data <- as.matrix(cbind(X, Y))
    variable.names <- c(colnames(X), "outcome")
    outcome.index <- ncol(input.data)
    
    forest <- regression_train(input.data, outcome.index, sparse.data,
        variable.names, mtry, num.trees, verbose, num.threads, min.node.size, sample.with.replacement,
        keep.inbag, sample.fraction, no.split.variables, seed, honesty, ci.group.size, alpha)
    
    forest[["ci.group.size"]] <- ci.group.size
    forest[["original.data"]] <- input.data
    forest[["feature.indices"]] <- 1:ncol(X)
    class(forest) <- c("regression_forest", "grf")
    forest
}

#' Predict with a regression forest
#' 
#' Gets estimates of E[Y|X=x] using a trained regression forest.
#'
#' @param object The trained forest.
#' @param newdata Points at which predictions should be made. If NULL,
#'                makes out-of-bag predictions on the training set instead
#'                (i.e., provides predictions at Xi using only trees that did
#'                not use the i-th training example).
#' @param num.threads Number of threads used in training. If set to NULL, the software
#'                    automatically selects an appropriate amount.
#' @param estimate.variance Whether variance estimates for hat{tau}(x) are desired
#'                          (for confidence intervals).
#' @param ... Additional arguments (currently ignored).
#'
#' @return Vector of predictions.
#' @export
predict.regression_forest <- function(object, newdata = NULL,
                                      num.threads = NULL,
                                      estimate.variance = FALSE,
                                      ...) {
    
    if (is.null(num.threads)) {
        num.threads <- 0
    } else if (!is.numeric(num.threads) | num.threads < 0) {
        stop("Error: Invalid value for num.threads")
    }
    
    sparse.data <- as.matrix(0)
    variable.names <- character(0)
    
    if (estimate.variance) {
        ci.group.size = object$ci.group.size
    } else {
        ci.group.size = 1
    }
    
    forest.short <- object[-which(names(object) == "original.data")]
    
    if (!is.null(newdata)) {
        input.data <- as.matrix(cbind(newdata, NA))
        regression_predict(forest.short, input.data, sparse.data, variable.names, 
                           num.threads, ci.group.size)
    } else {
        input.data <- object[["original.data"]]
        regression_predict_oob(forest.short, input.data, sparse.data, variable.names, 
                               num.threads, ci.group.size)
    }
}