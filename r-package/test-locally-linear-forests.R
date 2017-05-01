#locally.linear.forest <- function(X,Y,sample.fraction = 0.5, mtry = ceiling(ncol(X)/3), 
# num.trees = 500, num.threads = NULL, min.node.size = NULL, keep.inbag = FALSE, 
# honesty = TRUE, ci.group.size = 2, seed = NULL, lambda=0.01)

# data: assume X has column of 1's added
# recall LLF returns theta right now 

# first goal: find the picture to fix

library(gradient.forest)
library(Rcpp)
library(RcppEigen)

linear_tail_2 <- function(x){
  if(x<0.5){
    return(((x+1)**3/2))
  }else{
    return(sqrt(x)+(1.5)**3/2-sqrt(.5))
  }
}

n <- 300
x <- runif(n,-1,1)
y <- sapply(x, function(x_i){linear_tail_2(x_i)}) + rnorm(n,0,0.05)

jpeg("rawplot.jpg")
plt = plot(x,y)
dev.off()

lambda = 0.01

X <- cbind(1,x,rnorm(n,0,0.5))

forest_one <- locally.linear.forest(X,y,sample.fraction = 0.5, mtry = 2,min.node.size = 10, ci.group.size = 1, lambda=lambda)
weights <- predict(forest_one,newdata=data.frame(X)) # check if OOB prediction is working 

J <- diag(3)
J[1,1] <- 0

results <- sapply(1:n, function(i){
  weight_matrix <- diag(weights[i,])
  M_inverse <- solve( t(X) %*% weight_matrix %*% X + lambda*J)
  second_term <- t(X) %*% weight_matrix
  full_pred <- M_inverse %*% second_term %*% y
  t(X[i,]) %*% full_pred
})

df <- data.frame(cbind(x,results))
newdf <- df[order(x),]

jpeg("afterplot.jpg")
plt = plot(x,y)
points(newdf$x, newdf$results,col="red",'l',lwd=4)
dev.off()
