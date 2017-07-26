locally.linear.forest <- function(X,Y,sample.fraction = 0.5, mtry = ceiling(ncol(X)/3), variable_names = NULL, lambda = 1,
                                  num.trees = 500, num.threads = NULL, min.node.size = NULL, keep.inbag = FALSE, 
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
  
  input.data <- as.matrix(cbind(X, Y))
  #variable.names <- c(colnames(X), "outcome")
  variable_names <- c(variable_names, "outcome")
  outcome.index <- ncol(input.data)
  outcome.index.zeroindexed <- outcome.index - 1
  no.split.variables <- numeric(0)
  
  forest <- locally_linear_train(input.data, outcome.index.zeroindexed, sparse.data, variable_names,
                                 lambda, mtry, num.trees, verbose, num.threads, min.node.size, sample.with.replacement, 
                                 keep.inbag, sample.fraction, no.split.variables, seed, honesty, ci_group_size)
  
  forest[["original.data"]] <- input.data
  class(forest) <- "locally.linear.forest"
  forest
}

predict.locally.linear.forest <- function(forest, lambda=1, newdata = NULL, num.threads = NULL) {
  
  if (is.null(num.threads)) {
    num.threads <- 0
  } else if (!is.numeric(num.threads) | num.threads < 0) {
    stop("Error: Invalid value for num.threads")
  }
  
  sparse.data <- as.matrix(0)
  variable.names <- character(0)
  forest.short <- forest[-which(names(forest) == "original.data")]
  
  if (!is.null(newdata)) {
    training.data <- forest[["original.data"]]
    p <- ncol(training.data) - 1
    training.data <- training.data[,1:p] # remove responses, since they are already stored
    
    sparse.training <- as.matrix(0)
    input.data <- as.matrix(cbind(newdata, NA))
    
    #print("Cpp yhat")
    yhat <- locally_linear_predict(forest.short, input.data, sparse.data, training.data, sparse.training, lambda, variable.names, num.threads)
    #yhat <- sapply(1:n, function(i){
    #w = weights[i,]
    #fit = glmnet(X,Y,weights=w,alpha=0,lambda=lambda)
    #predict(fit, newdata)[i]
    #})
    #print("printing R yhat")
    #print(yhat)
    return(yhat)
    
  } else {
    print("oob prediction")
    
    input.data <- forest[["original.data"]]
    
    print("found input data, calling cpp predict_oob now")
    
    print("cpp yhat")
    yhat <- locally_linear_predict_oob(forest.short, input.data, sparse.data, lambda, variable.names, num.threads)
    #weights <- locally_linear_predict_oob(forest.short, input.data, sparse.data, variable.names,num.threads)
    #yhat <- 0
    #print("not yet implemented")
    
    #yhat <- sapply(1:n, function(i){
    #  w = weights[i,]
    #  fit = glmnet(X,Y,weights=w,alpha=0,lambda=lambda)
    #  predict(fit)[i]
    #})
    print("printing R yhat")
    print(yhat)
    return(yhat)               
  }
}
