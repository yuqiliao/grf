rm(list = ls())
library(gradient.forest)

p = 10
n = 600
frac = 0.70
mtry = 4
sigma = 0.1
names = c("yay", "wooo")
mu = function(xx) log(1 + exp(6 * xx))
leaf.size = 1
lambda = 0.5

X = matrix(runif(p*n, -1, 1), n, p)
Y = sapply(X[,1], mu) + sigma * rnorm(n)
llf = locally.linear.forest(X, Y, sample.fraction = frac, mtry = mtry, variable_names = names, lambda = lambda, 
                            num.trees = 500, min.node.size = leaf.size, ci_group_size = 1)
yhat = predict(llf)
error = sum((yhat-Y)**2)/n
xx = seq(-1,1,length=200)

setwd("~/Desktop")
png(filename="boundary.png")
plot(X[,1],yhat, main="Locally Linear Forest",ylim=c(0,6),xlab="x",ylab="y")
lines(xx,mu(xx),lwd=2,col=2)
text(x=-0.5, y=1, labels=paste("error=",as.character(round(error,4))))
dev.off()