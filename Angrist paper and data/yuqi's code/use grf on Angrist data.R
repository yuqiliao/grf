#### Try to use r grf package on Angrist data to replicate the results in the Athey paper (Figure 6 on page 35)
#### Yuqi Liao
#### 2/16/18

# Setting things up
#install.packages("grf")
rm(list = ls())
library("grf")
library("haven")
library("dplyr")
library("AER")


# Read in data
# Note that in the Angrist paper/data constructs two datasets both using Census 1980 5% PUMS data: one for all women with two or more children (n = 394840); the other for all *married* women with two or more children (n = 254652). However, in the Athey paper, it studys "a sample of n = 334535 married mothers with at least 2 children (1980 census data)". **I cannot find ways to come up with n = 334535**
# Read in dataset for married women with at least 2 children (n = 254652)
data <- read_sas("G:/Data Science/Generalized Random Forest/Angrist paper and data/data/AngEv98/subset/twob.sas7bdat")


# Define variables
X <- data %>% select(agem1, agefstm, educm, incomed, blackm, hispm, othracem)    # Note that the Athey paper and the Angrist paper have slightly different covariates. Here's a label of the covariates used in the Athey paper.
    ## data$agem1 <age in years of mom>
    ## data$agefstm <age of mom at first birth>
    ## data$educm <education, mother>
    ## data$incomed <labor income, father>
    ## data$blackm <black, mother>
    ## data$hispm <hispanic, mother>
    ## data$othracem <other race, mother>

    ### The Angrist paper didn't include data$educm nor data$incomed, but it includes the following two variables
    # data$boy1st <first birth boy>
    # data$boy2nd <second birth boy>
Y <- data$workedm #The outcome Yi is whether the mother did not work in the year preceding the census
W <- data$morekids  #the treatment Wi is whether the mother had 3 or more children at census time
Z <- 1 - data$samesex # "samesex" here stands for mothers whose first two kids were of same sexes. the instrument Zi measures whether or not the mother's first two children were of different sexes. 


# Calculate covariations
cov(W, Z) # covariation between W and Z is around -0.0167, which is around the same by the Athey paper (-0.016)
cov(Y, Z) # covariation between Y and Z is around 0.00226, which is around the same by the Athey paper (0.0021)


# Use ivreg() to get the local average treatment effect
# **YL: Note that, the Athey paper says the ivreg treatment effect is 0.14 +- 0.054. I don't seem to be able to replicate it in the following two models. The coeffcient on W is around -1.2 or -1.3 in my results below**
# With covariates
regular.iv.1 <- ivreg(Y ~ W + X$agem1 + X$agefstm + X$educm + X$incomed + X$blackm + X$hispm + X$othracem | Z + X$agem1 + X$agefstm + X$educm + X$incomed + X$blackm + X$hispm + X$othracem)
summary(regular.iv.1)
confint(regular.iv.1, level = 0.95)
# Without covariates
regular.iv.2 <- ivreg(Y ~ W | Z)
summary(regular.iv.2)
confint(regular.iv.2, level = 0.95)


# Train the dataset with instrumental_forest() #*Note that in the paper min.node.size = 800, and num.trees = 100000.* Such number of trees will freeze my computer when trying to run it.
forest.iv <- instrumental_forest(X, Y, W, Z, sample.fraction = 0.05, 
                                 #min.node.size = 800, 
                                 num.trees = 1000, ci.group.size = 2)


# Create a test dataset. According to the Athey paper, "We vary the mother's age at first birth and the father's income; other covariates are set to their median values in the above plots."
X$median.agem1 <- median(X$agem1) # could write a loop to simpify these lines
X$median.educm <- median(X$educm)
X$median.blackm <- median(X$educm)
X$median.hispm <- median(X$educm)
X$median.othracem <- median(X$educm)
# Test dataset where mothers are 18 years old at first birth 
X.test.agefstm18 <- X %>% 
            select (incomed, starts_with("median"), agefstm) %>% 
            filter (agefstm == 18)
# Test dataset where mothers are 22 years old at first birth 
X.test.agefstm22 <- X %>% 
  select (incomed, starts_with("median"), agefstm) %>% 
  filter (agefstm == 22)


# Predict using the forest **Note that none of the predicted results have many variations, indicating that something is not right before this step**
preds.iv.18 <- predict(forest.iv, X.test.agefstm18, estimate.variance = TRUE)$predictions
preds.iv.22 <- predict(forest.iv, X.test.agefstm22, estimate.variance = TRUE)$predictions


# Plot the result. Try to replicate Figure 6 on page 35 of the Athey paper. **Accordingly, the plot shows that almost all dots have the same Y, which is puzzling**
x_axis.18 <- as.numeric(unlist(X.test.agefstm18[,1]))
plot(x_axis.18, preds.iv.18, ylim = range(preds.iv.18), xlab = "Father's Income [$1k/year]", ylab = "Y", type = "p")

x_axis.22 <- as.numeric(unlist(X.test.agefstm22[,1]))
plot(x_axis.22, preds.iv.22, ylim = range(preds.iv.22), xlab = "Father's Income [$1k/year]", ylab = "Y", type = "p")


# Calculate and add confidence intervals
sigma.hat.18 = sqrt(predict(forest.iv, X.test.agefstm18, estimate.variance = TRUE)$variance.estimates)
plot(x_axis.18, preds.iv.18, ylim = range(preds.iv.18 + 1.96 * sigma.hat.18, preds.iv.18 - 1.96 * sigma.hat.18), xlab = "x", ylab = "Y", type = "l")
lines(x_axis.18, preds.iv.18 + 1.96 * sigma.hat.18, col = 1, lty = 2)
lines(x_axis.18, preds.iv.18 - 1.96 * sigma.hat.18, col = 1, lty = 2)


#### The code below only works if causal_forest() is used
# Train the dataset with causal_forest() and compare the results with that trained by instrumental_forest(). 
# In the Athey paper 7.2, it talks about how instrumental_forest() works better in some situations
forest.causal <- causal_forest(X, Y, W, sample.fraction = 0.05, 
                                 #min.node.size = 800, 
                                 num.trees = 1000, ci.group.size = 2)
preds.causal.18 <- predict(forest.causal, X.test.agefstm18, estimate.variance = TRUE)$predictions

x_axis.18 <- as.numeric(unlist(X.test.agefstm18[,1]))
plot(x_axis.18, preds.causal.18, ylim = range(preds.causal.18), xlab = "Father's Income [$1k/year]", ylab = "Y", type = "p")

# Estimate the conditional average treatment effect on the full sample (CATE).
estimate_average_effect(forest.causal, target.sample = "all") 

# Estimate the conditional average treatment effect on the treated sample (CATT).
estimate_average_effect(forest.causal, target.sample = "treated")




