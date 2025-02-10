#' Dr. Eduardo Vinicius da Silva Oliveira
#' PIBILab  - UFS
#' https://github.com/PIBILab/intro-ml-for-ecology
##################################################################
#' Boosting and bagging

# -------------------------------------------------------------------------
# Packages

install.packages("xgboost")
install.packages("caret")
install.packages("Matrix")
install.packages("randomForest")
install.packages("pROC")
install.packages("openxlsx")
library(xgboost)
library(caret)
library(Matrix)
library(randomForest)
library(pROC)
library(openxlsx)

# -------------------------------------------------------------------------

###
#' 1-Boosting

# Input data 

occ_data<-read.xlsx("https://github.com/PIBILab/intro-ml-for-ecology/raw/main/4-Species_distribution_modelling/data/occurrence.xlsx")

vars<-read.xlsx("https://github.com/PIBILab/intro-ml-for-ecology/raw/main/4-Species_distribution_modelling/data/climatic_data.xlsx")

# Train and test data
presence <- occ_data$presence
trainIndex <- createDataPartition(presence, p = 0.8, list = FALSE)
data_train <- occ_data[trainIndex, ]
data_test <- occ_data[-trainIndex, ]

# Environmental variables
train_envi <- vars[trainIndex, ]
test_envi <- vars[-trainIndex, ]

# Setting data for XGBoost
train_data <- xgb.DMatrix(data = as.matrix(cbind(data_train[, c("x", "y")], train_envi)), label = data_train$presence)
test_data <- xgb.DMatrix(data = as.matrix(cbind(data_test[, c("x", "y")], test_envi)), label = data_test$presence)

# Train the XGBoost model
param <- list(
  objective = "binary:logistic",  
  eval_metric = "logloss"         
)

xgb_model <- xgboost(
  data = train_data, 
  params = param, 
  nrounds = 100,  # numbers of iterations
  verbose = 1
)

# Predicting with the XGBoost model
pred_xgb <- predict(xgb_model, test_data)

# Evaluation (AUC)
roc_curve <- roc(data_test$presence, pred_xgb)
auc(roc_curve)

###
#' 2-Bagging

# Train and test data
presence <- occ_data$presence
trainIndex <- createDataPartition(presence, p = 0.8, list = FALSE)
data_train <- occ_data[trainIndex, ]
data_test <- occ_data[-trainIndex, ]

# Environmental variables
train_envi <- vars[trainIndex, ]
test_envi <- vars[-trainIndex, ]

# Training the Random Forest model (Bagging)
rf_model <- randomForest(
  presence ~ ., 
  data = cbind(data_train, train_envi), 
  ntree = 500  # number of trees
)

# Predicting with the Random Forest model
pred_rf <- predict(rf_model, newdata = cbind(data_test, test_envi))

# Evaluation (AUC)
roc_curve_rf <- roc(data_test$presence, pred_rf)
auc(roc_curve_rf)

rm(list = ls(all.names = TRUE)) #Clear all objects 
gc() #Report the memory usage
# end
######################################################################################################################################
