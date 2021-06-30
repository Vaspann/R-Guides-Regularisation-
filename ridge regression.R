#Ridge regression is a method of estimating the coefficients of multiple-regression 
#models in scenarios where independent variables are highly correlated. 
#In addition it reduces variance by incorporating bias, whcih can lead to lower errors



#Load the packages
library(glmnet)
library(psych)
library(caTools)
library(caret)
library(ggplot2)

#Load the data
data(package = "ggplot2")
data("mpg")

# Select features for analyses 
df <- mpg[,c("displ","cyl", "cty", "hwy", "fl")]

#Inspect the data
summary(df)
str(df)

#Change classes
df$cyl <- as.factor(df$cyl)
df$fl <- as.factor(df$fl)

#Check for multicollinearity
#One common cut-off for collinearity is 0.8 
pairs.panels(df[,-1])


# train and test set ------------------------------------------------------

set.seed(123)
sep <- sample(2,nrow(df),replace=T,prob = c(0.7,0.3))
train <- df[sep==1,]
test <- df[sep==2,]



# Models ------------------------------------------------------------------



#Custom control parameters
custom <- trainControl(method = "repeatedcv",
                       number = 10,
                       repeats = 5)

# Genaral Linear Model
set.seed(123)
glm_model <- train(displ~.,
            train,
            method = "glm",
            trControl = custom)

#results
glm_model$results
glm_model
summary(glm_model)
plot(glm_model$finalModel)
plot(varImp(glm_model, scale = F))



# Ridge -------------------------------------------------------------------

set.seed(123)
ridge_model <- train(displ~.,
               train,
               method = "glmnet",
               tuneGrid = expand.grid(alpha = 0,
                                      lambda = seq(0.0001, 1, length=5)),
               trControl = custom)

#results
ridge_model
plot(ridge_model)
plot(ridge_model$finalModel, xvar = "lambda", label = T)
plot(ridge_model$finalModel, xvar = "dev", label = T)
plot(varImp(ridge_model, scale = F))



# Compare Models ----------------------------------------------------------

mod_list <- list(GLM = glm_model, RIDGE = ridge_model)

res <- resamples(mod_list)

#In this example implementing ridge regularisation did not improve the model 
summary(res)

bwplot(res)

xyplot(res, metric = "MAE")
xyplot(res, metric = "RMSE")



#Predictions

pred_test <- predict.train(glm_model,data = test)
sqrt(mean((test$displ-pred_test)^2)) 

