rm(list = ls())
library(gradient.forest)

mu = function(xx) log(1 + exp(6 * xx))

errors = function(p, n = 600, frac = 0.70, mtry = ceiling(p/3), sigma = 0.1, leaf.size = 1, lambda = 1, 
                  numreps = 10, cutoff = 0.8, standardize = TRUE){
  mses = rep(-1, numreps)
  tail.mses = rep(-1, numreps)
  for(i in 1:numreps){
    X = matrix(runif(p*n, -1, 1), n, p)
    Y = sapply(X[,1], mu) + sigma * rnorm(n)
    
    if(standardize){
      X = scale(X)
    }
    
    llf = locally.linear.forest(X, Y - mean(Y), sample.fraction = frac, mtry = mtry,lambda = lambda, 
                                num.trees = 500, min.node.size = leaf.size, ci_group_size = 1)
    
    newX = matrix(runif(p*n, -1, 1), n, p)
    newY = sapply(newX[,1], mu) + sigma * rnorm(n)
    
    if(standardize){
      newtemp = predict(llf, newdata=scale(newX)) + mean(Y)
    }else{
      newtemp = predict(llf, newdata = newX) + mean(Y)
    }
    
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

## Examine Effect of Regularization ##

lambdas = seq(0,10,length=41)
test.lambda = sapply(lambdas, function(lambda){
  errors(p=10,lambda=lambda)[1]
})
plot(lambdas, test.lambda, 'l', xlab = "lambda", ylab="MSE", main = "MSE by Regularization (unscaled)")

#################################################
## Compare Effects of Standardizing and Lambda ##
#################################################

lambdas = seq(0,10,length=41)

reg.lambda = sapply(lambdas, function(lambda){errors(p=10,lambda=lambda, standardize = TRUE)[1]})
unreg.lambda = sapply(lambdas, function(lambda){errors(p=10,lambda=lambda, standardize = FALSE)[1]})

plot(lambdas, reg.lambda, 'l', lwd = 2, ylim = range(reg.lambda, unreg.lambda), xlab = "lambda", ylab="MSE", main = "MSE by Regularization")
lines(lambdas, unreg.lambda, 'l', col="blue", lwd=2)
legend("topleft", c("Standardized", "Unstandardized"), col=c("black","blue"), cex=0.5, lwd=c(2,2))