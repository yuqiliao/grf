rm(list = ls())
library(gradient.forest)

p = 10
n = 600
frac = 0.70
mtry = 4
sigma = 0.1
leaf.size = 2
lambda = 2.5

mu = function(xx) log(1 + exp(6 * xx))
X = matrix(runif(p*n, -1, 1), n, p)
Y = sapply(X[,1], mu) + sigma * rnorm(n)

X = scale(X)

llf = locally.linear.forest(X, Y - mean(Y), sample.fraction = frac, mtry = mtry, lambda = lambda, 
                            num.trees = 500, min.node.size = leaf.size, ci_group_size = 1)
yhat = predict(llf) + mean(Y)
error = sum((yhat-Y)**2)/n
xx = seq(-1,1,length=200)

X = X * attr(X, 'scaled:scale') + attr(X, 'scaled:center')

#setwd("~/Desktop")
#png(filename="boundary.png")
plot(X[,1],yhat, main="",ylim=c(0,6),xlab="x",ylab="y")
lines(xx,mu(xx),lwd=2,col=2)
text(x=-0.8, y=1, labels=paste("error=",as.character(round(error,4))), cex=0.6)
#dev.off()