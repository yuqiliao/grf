rm(list = ls())
library(gradient.forest)

p = 10
n = 600
frac = 0.70
mtry = 4
sigma = 0.1
names = c("yay", "wooo")
mu = function(xx) log(1 + exp(6 * xx))
leaf.size = 5
#lambda = 1 

# prediction error test
grid = sapply(c(0,3,5,7,10,15), function(lambda){
  prederrors = sapply(1:5, function(i){
    X = matrix(runif(p*n, -1, 1), n, p)
    Y = sapply(X[,1], mu) + sigma * rnorm(n)
    
    llf = locally.linear.forest(X, Y, sample.fraction = frac, mtry = mtry, variable_names = names, 
                                lambda = lambda, num.trees = 500, min.node.size = leaf.size, ci_group_size = 1)
    
    newX = matrix(runif(p*n, -1, 1), n, p)
    newY = sapply(newX[,1], mu) + sigma * rnorm(n)
    
    newtemp = predict(llf, newdata=newX)
    sum((newtemp-newY)**2)/n
  })
  
  print(paste("average prediction error for lambda = ", lambda, ", leaf size = ", leaf.size, ", p = ", p, ", and sigma = ", sigma))
  print(mean(prederrors))
})

#prederrors = sapply(1:5, function(i){
#  X = matrix(runif(p*n, -1, 1), n, p)
#  Y = sapply(X[,1], mu) + sigma * rnorm(n)
#  llf = locally.linear.forest(X, Y, sample.fraction = frac, mtry = mtry, variable_names = names, 
#                              lambda = lambda, num.trees = 500, min.node.size = leaf.size, ci_group_size = 1)
#  newX = matrix(runif(p*n, -1, 1), n, p)
#  newY = sapply(newX[,1], mu) + sigma * rnorm(n)
#  newtemp = predict(llf, newdata=newX)
##  sum((newtemp-newY)**2)/n
#})
#print(paste("average prediction error for lambda = ", lambda, ", leaf size = ", leaf.size, ", p = ", p, ", and sigma = ", sigma))
#print(mean(prederrors))