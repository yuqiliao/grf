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

n<- 300
x <- runif(n,-1,1)
y <- sapply(x, function(x_i){linear_tail_2(x_i)}) + rnorm(n,0,0.05)
#plt = plot(x,y)


#jpeg("myplot.jpg")
#plt = plot(x,y)
#dev.off()

X <- cbind(rep(1,n),x,rnorm(n,0,1))

quick_new_point <- c(1,0.4,-0.3)

forest_one <- locally.linear.forest(X,y,sample.fraction = 0.5, mtry = 2,num.trees = 2, num.threads = NULL, min.node.size = NULL, keep.inbag = FALSE, honesty = TRUE, ci.group.size = 2, seed = NULL, lambda=0.01)

test_pred <- predict(forest_one, newdata=quick_new_point, lambda=0.01)
#full_pred <- predict(forest_one,newdata=X,lambda=0.01)

#jpeg("afterplot.jpg")
#plt = plot(x,full_pred)
#dev.off()
