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
#data <- read_sas("G:/Data Science/Generalized Random Forest/Angrist paper and data/data/AngEv98/subset/twob.sas7bdat")
data <- read_sas("/Users/Yuqi/Google Drive/AIR/Data Science/GRF/Angrist paper and data/data/twob.sas7bdat")


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
#Y2 <- data$workedm #The outcome Yi is whether the mother work in the year preceding the census
Y <- 1 - data$workedm #The outcome Yi is whether the mother DID NOT work in the year preceding the census
W <- data$morekids  #the treatment Wi is whether the mother had 3 or more children at census time
Z <- data$samesex # "samesex" here stands for mothers whose first two kids were of same sexes. the instrument Zi measures whether or not the mother's first two children were of same sexes. 
#Z2 <- 1 - data$samesex # "samesex" here stands for mothers whose first two kids were of same sexes. the instrument Zi measures whether or not the mother's first two children were of different sexes. 


# Calculate covariations
cov(W, Z) # covariation between W and Z is around 0.0167, which is around the same by the Athey paper (0.016)
cov(Y, Z) # covariation between Y2 and Z is around 0.00226, which is around the same by the Athey paper (0.0021)


# Use ivreg() to get the local average treatment effect
# **YL: Note that, the Athey paper says the ivreg treatment effect is 0.14 +- 0.054 = [0.086, 0.194] . My estimate is close: The first model [0.066, 0.175], the second model [0.077, 0.193]
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
                                 num.trees = 2000, ci.group.size = 2)


# Create the test dataset
preds.iv <- predict(forest.iv, estimate.variance = TRUE) #according to ?predict.instrumental_forest, newdata is set to NULL by default. "If NULL, makes out-of-bag predictions on the training set instead (i.e., provides predictions at Xi using only trees that did not use the i-th training example)". So I think here the function will automatically uses the 95% of the dataset to predict results.

#Calculate predicted treatment effect
preds <- preds.iv$predictions
# Calculate sigma.hat
sigma.hat = sqrt(preds.iv$variance.estimate)
preds.with.ci <- cbind(preds, preds - 1.96 * sigma.hat, preds + 1.96 * sigma.hat)


# Plotting
# According to the Athey paper, "We vary the mother's age at first birth and the father's income; other covariates are set to their median values in the above plots."
# create a new dataset for plotting where I can filter by mother's age at first birth

df <- X
df$preds <- preds
df$lowerci <- preds - 1.96 * sigma.hat
df$higherci <- preds + 1.96 * sigma.hat

df.18 <- df %>% filter(agefstm == 18)

plot(df.18$incomed, df.18$preds, xlim = range(df.18$incomed), ylim = range(df.18$preds), xlab = "Father's Income ", ylab = "Y")
lines(df.18$incomed, df.18$preds, lwd = 1, col = 2)
lines(df.18$incomed, df.18$lowerci, lwd = 1, col = 1, lty = 2)
lines(df.18$incomed, df.18$higherci, lwd = 1, col = 1, lty = 2)














# Create a test dataset. 
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




