rm(list = ls())
library(gradient.forest)

errors = function(n = 600, p = 10, frac = 0.70, mtry = ceiling(p/3), sigma = 0.1, leaf.size = 1, lambda = 1, numreps = 10, cutoff = 0.8){
  mses = rep(-1, numreps)
  tail.mses = rep(-1, numreps)
  for(i in 1:numreps){
    X = matrix(runif(p*n, -1, 1), n, p)
    Y = sapply(X[,1], mu) + sigma * rnorm(n)
    
    llf = locally.linear.forest(X, Y, sample.fraction = frac, mtry = mtry,lambda = lambda, 
                                num.trees = 500, min.node.size = leaf.size, ci_group_size = 1)
    
    newX = matrix(runif(p*n, -1, 1), n, p)
    newY = sapply(newX[,1], mu) + sigma * rnorm(n)
    
    newtemp = predict(llf, newdata=newX)
    mses[i] = sum((newtemp-newY)**2)/n
    
    temp = 0
    temp.len = 0
    for(j in 1:n){
      if(newX[j,1] > cutoff){
        temp = temp + (newtemp[j] - newY[j])**2 
        temp.len = temp.len + 1
      }
    }
    temp = temp/temp.len
    tail.mses[i] = temp
  }
  return(c(mean(mses), mean(tail.mses)))
}

lambdas = seq(0,5,length=20)
test.lambda = sapply(lambdas, function(lambda){
  errors(lambda=lambda, numreps=20)[1]
})
plot(lambdas, test.lambda, 'l', xlab = "lambda", ylab="MSE", main = "MSE by Regularization")