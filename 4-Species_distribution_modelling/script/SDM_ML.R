#' ---
# Title: R Codes for species distribution modeling
# https://github.com/PIBILab/intro-ml-for-ecology
# by Dr. Eduardo V. S. Oliveira
# 04/02/2025
#' ---

# Species distribution modeling from Machine Learning---------

## Libraries

if(!require(randomForest)) install.packages("randomForest")
if(!require(e1071)) install.packages("e1071")
if(!require(gbm)) install.packages("gbm")
if(!require(caret)) install.packages("caret")
if(!require(dismo)) install.packages("dismo")
if(!require(xgboost)) install.packages("xgboost")
if(!require(nnet)) install.packages("nnet")
if(!require(openxlsx)) install.packages("openxlsx")

# -------------------------------------------------------------------------

## Input data 

data1<-read.xlsx("https://github.com/PIBILab/intro-ml-for-ecology/raw/main/1-Decision_trees/data/data_urb_affor.xlsx")

occ_data<-read.xlsx("occurrence.xlsx")
vars<-read.xlsx("climatic_data.xlsx")

## Train and test data
presence <- occ_data$presence
trainIndex <- createDataPartition(presence, p = 0.8, list = FALSE)
data_train <- occ_data[trainIndex, ]
data_test <- occ_data[-trainIndex, ]

train_envi <- vars[trainIndex, ]
test_envi <- vars[-trainIndex, ]

set_data = cbind(data_train, train_envi)

## Creating and training the machine learning models

# Random Forest
rf_model <- randomForest(presence ~ ., data = set_data, ntree = 500)

# SVM
svm_model <- svm(presence ~ ., data = set_data, probability = TRUE)

# Gradient Boosting 
gbm_model <- gbm(presence ~ ., data = set_data, distribution = "bernoulli", n.trees = 500)

# K-Nearest Neighbors 
knn_model <- train(presence ~ ., data = set_data, method = "knn", trControl = trainControl(method = "cv", number = 10))

# Logistic regression (GLM)
logi_model <- glm(presence ~ ., data = set_data, family = "binomial")

# XGBoost
xgb_data <- xgb.DMatrix(data = as.matrix(cbind(data_train[, c("x", "y")], train_envi)), label = data_train$presence)

xgb_model <- xgboost(data = xgb_data, nrounds = 500, objective = "binary:logistic")

# Neural networks (MLP)
mlp_model <- nnet(presence ~ ., data = set_data, size = 5, linout = TRUE)

## Model predictions

# Random Forest
rf_pred <- predict(rf_model, newdata = cbind(data_test, test_envi), type = "response")

# SVM
svm_pred <- predict(svm_model, newdata = cbind(data_test, test_envi), probability = TRUE)
svm_prob <- attr(svm_pred, "probabilities")[, 2]  

# GBM
gbm_pred <- predict(gbm_model, newdata = cbind(data_test, test_envi), type = "response", n.trees = 500)

# KNN
knn_pred <- predict(knn_model, newdata = cbind(data_test, test_envi))

# Logistic regression
logistic_pred <- predict(logi_model, newdata = cbind(data_test, test_envi), type = "response")

# XGBoost
xgb_pred <- predict(xgb_model, newdata = as.matrix(cbind(data_test[, c("x", "y")], test_envi)))

# MLP
mlp_pred <- predict(mlp_model, newdata = cbind(data_test, test_envi))

## Evaluating the models with train data

# Function to calculate the AUC
calc_auc <- function(pred, incidence) {
  library(pROC)
  roc_curve <- roc(incidence, pred)
  return(auc(roc_curve))
}

# AUC for Random Forest
auc_rf <- calc_auc(rf_pred, data_test$presence)

# AUC for SVM
auc_svm <- calc_auc(svm_pred, data_test$presence)

# AUC for GBM
auc_gbm <- calc_auc(gbm_pred, data_test$presence)

# AUC for KNN
auc_knn <- calc_auc(knn_pred, data_test$presence)

# AUC for logistic regression
auc_logistic <- calc_auc(logistic_pred, data_test$presence)

# AUC for XGBoost
auc_xgb <- calc_auc(xgb_pred, data_test$presence)

# AUC para MLP
auc_mlp <- calc_auc(mlp_pred, data_test$presence)

# Show the results
print(paste("AUC for Random Forest: ", auc_rf))
print(paste("AUC for SVM: ", auc_svm))
print(paste("AUC for GBM: ", auc_gbm))
print(paste("AUC for KNN: ", auc_knn))
print(paste("AUC for Regressão Logística: ", auc_logistic))
print(paste("AUC for XGBoost: ", auc_xgb))
print(paste("AUC for MLP: ", auc_mlp))

rm(list = ls(all.names = TRUE)) #Clear all objects 
gc() #Report the memory usage
# end
######################################################################################################################################
