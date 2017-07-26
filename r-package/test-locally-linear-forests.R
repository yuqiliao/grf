library(gradient.forest)
library(Rcpp)
library(RcppEigen)

mu = function(xx) log(1 + exp(6 * xx))
n = 50
p = 4
mtry = 2
frac = 0.7
nodesize = 3
lambda = 1

X = matrix(runif(p*n, -1, 1), n, p)
Y = sapply(X[,1], mu) + 0.1 * rnorm(n)
names = c("x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10")

llf = locally.linear.forest(X, Y, sample.fraction = frac, mtry = mtry, variable_names = names,
                            lambda = 1, num.trees = 500, num.threads = NULL, min.node.size = NULL, 
                            keep.inbag = FALSE, honesty = TRUE, ci_group_size = 1, seed = NULL)

X.new = matrix(runif(p*n, -1, 1), n, p)
Y.new = sapply(X.new[,1], mu) + 0.1 * rnorm(n)

yhat = predict(llf,newdata=X.new)
err = sum((yhat-Y.new)**2)/n
print("error:")
print(err)