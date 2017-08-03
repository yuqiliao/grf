locally.linear.forest <- function(X,Y,sample.fraction = 0.5, mtry = ceiling(ncol(X)/3), variable_names = NULL, lambda = NULL,
                                  penalties = NULL, num.trees = 500, num.threads = NULL, min.node.size = NULL, keep.inbag = FALSE, 
                                  honesty = TRUE, ci_group_size = 1, seed = NULL) {
  sparse.data <- as.matrix(0)
  
  if (is.null(mtry)) {
    mtry <- 0
  } else if (!is.numeric(mtry) | mtry < 0) {
    stop("Error: Invalid value for mtry")
  }
  
  verbose = FALSE
  
  if (is.null(num.threads)) {
    num.threads <- 0
  } else if (!is.numeric(num.threads) | num.threads < 0) {
    stop("Error: Invalid value for num.threads")
  }
  
  if (is.null(min.node.size)) {
    min.node.size <- 0
  } else if (!is.numeric(min.node.size) | min.node.size < 0) {
    stop("Error: Invalid value for min.node.size")
  }
  
  sample.with.replacement <- FALSE
  
  if (!is.logical(keep.inbag)) {
    stop("Error: Invalid value for keep.inbag")
  }
  
  if (!is.numeric(sample.fraction) | sample.fraction <= 0 | sample.fraction > 1) {
    stop("Error: Invalid value for sample.fraction. Please give a value in (0,1].")
  }
  
  if (is.null(seed)) {
    seed <- runif(1, 0, .Machine$integer.max)
  }
  
  p <- ncol(X)
  
  if(is.null(penalties) & is.null(lambda)){
    regularization <- rep(1, p)
  }else if(is.null(lambda)){
    if(length(penalties) == p){
      regularization <- penalties 
    }else{
      stop("Error: Invalid length for penalties.")
    }
  }else if(is.null(penalties)){
    if(length(lambda) != 1){
      stop("Error: Invalid value for lamdba.")
    }
    regularization <- rep(lambda, p)
  }else{
    stop("Error: Invalid entries for regularization parameters. Use only lambda or penalties.")
  }
  
  input.data <- as.matrix(cbind(X, Y))
  variable_names <- c(variable_names, "outcome")
  outcome.index <- ncol(input.data)
  outcome.index.zeroindexed <- outcome.index - 1
  no.split.variables <- numeric(0)
  
  forest <- locally_linear_train(input.data, outcome.index.zeroindexed, sparse.data, variable_names,
                                 regularization, mtry, num.trees, verbose, num.threads, min.node.size, sample.with.replacement, 
                                 keep.inbag, sample.fraction, no.split.variables, seed, honesty, ci_group_size)
  
  forest[["original.data"]] <- input.data
  forest[["regularization"]] <- regularization
  class(forest) <- "locally.linear.forest"
  forest
}

predict.locally.linear.forest <- function(forest, newdata = NULL, num.threads = NULL) {
  
  if (is.null(num.threads)) {
    num.threads <- 0
  } else if (!is.numeric(num.threads) | num.threads < 0) {
    stop("Error: Invalid value for num.threads")
  }
  
  sparse.data <- as.matrix(0)
  variable.names <- character(0)
  forest.short <- forest[-which(names(forest) == "original.data")]
  
  regularization <- forest[["regularization"]]
  
  training.data <- forest[["original.data"]]
  p <- ncol(training.data) - 1
  training.data <- training.data[,1:p] 
  
  sparse.training <- as.matrix(0)
  
  if (!is.null(newdata)) {
    input.data <- as.matrix(newdata)
    
    yhat <- locally_linear_predict(forest.short, input.data, sparse.data, training.data, sparse.training, regularization, variable.names, num.threads)
    return(yhat)
    
  } else {
    input.data <- training.data
    
    yhat <- locally_linear_predict_oob(forest.short, input.data, sparse.data, training.data, sparse.training, regularization, variable.names, num.threads)
    return(yhat)               
  }
}
