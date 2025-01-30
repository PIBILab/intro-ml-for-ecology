#' Dr. Eduardo Vinicius da Silva Oliveira
#' PIBILab  - UFS
#' https://github.com/PIBILab/intro-ml-for-ecology
##################################################################
#' Compute decision tree in Machine Learning

# -------------------------------------------------------------------------
# Packages

install.packages("caret") 
install.packages("e1071")  
install.packages("rpart")       
install.packages("randomForest")
install.packages("e1071")       
install.packages("kernlab")     
install.packages("ggplot2")     
install.packages("gbm")         
library(caret)
library(e1071)
library(rpart)
library(randomForest)
library(e1071)
library(kernlab)
library(ggplot2)
library(gbm)

# -------------------------------------------------------------------------

###
#' Classification Tree 

# Input data 
data1<-read.xlsx("https://github.com/PIBILab/intro-ml-for-ecology/raw/main/1-Decision_trees/data/data_urb_affor.xlsx")

# Split the data (70% training, 30% testing)
trainIndex <- createDataPartition(data1$state, p = 0.7, list = FALSE)
trainData <- data1[trainIndex, ]
testData <- data1[-trainIndex, ]

# Control parameters for train
trainControl <- trainControl(
  method = "cv",  # Cross-validation
  number = 5     # Number of folds
)

# Train classification models

# 1. Logistic Regression
logisticModel <- train(
  state ~ ., 
  data = trainData, 
  method = "multinom",  
  trControl = trainControl
)

# 2. Random Forest
rfModel <- train(
  state ~ ., 
  data = trainData, 
  method = "rf",  
  trControl = trainControl
)

# 3. Support Vector Machine 
svmModel <- train(
  state ~ ., 
  data = trainData, 
  method = "svmRadial",  
  trControl = trainControl
)

# 4. k-Nearest Neighbors 
knnModel <- train(
  state ~ ., 
  data = trainData, 
  method = "knn",  
  trControl = trainControl
)

# 5. Gradient Boosting Machine 
gbmModel <- train(
  state ~ ., 
  data = trainData, 
  method = "gbm",  
  trControl = trainControl,
  verbose = FALSE
)

# Evaluate models 
models <- list(
  LogisticRegression = logisticModel,
  RandomForest = rfModel,
  SVM = svmModel,
  kNN = knnModel,
  GBM = gbmModel
)

resu1 <- resamples(models)
summary(resu1)

bwplot(resu1)

# Make predictions for a specific model 
pred1 <- predict(rfModel, newdata = testData)

# Confusion matrix and accuracy for Random Forest
c<-confusionMatrix(pred1, as.factor(testData$state))

###
#' Regression Tree 

# Split the data (70% training, 30% testing)
trainIndex <- createDataPartition(data1$DBH_cm, p = 0.7, list = FALSE)
trainData <- data1[trainIndex, ]
testData <- data1[-trainIndex, ]

# Control parameters for train
trainControl <- trainControl(
  method = "cv",  # Cross-validation
  number = 5,     # Number of folds
  summaryFunction = defaultSummary     
)

# Train regression models

# 1. Decision Tree Regression
dtModel <- train(
  DBH_cm ~ ., 
  data = trainData, 
  method = "rpart",  
  trControl = trainControl,
  tuneLength = 5)

# 2. Random Forest Regression
rfModel <- train(
  DBH_cm ~ ., 
  data = trainData, 
  method = "rf",  # Random Forest
  trControl = trainControl,
  tuneLength = 5  # Number of mtry values to try
)

# 3. Support Vector Regression 
svmModel <- train(
  DBH_cm ~ ., 
  data = trainData, 
  method = "svmRadial",  # SVM with radial kernel
  trControl = trainControl,
  tuneLength = 5         # Number of cost and sigma values to try
)

# 4. k-Nearest Neighbors Regression
knnModel <- train(
  DBH_cm ~ ., 
  data = trainData, 
  method = "knn",  # k-Nearest Neighbors
  trControl = trainControl,
  tuneLength = 5   # Number of k values to try
)

# 5. Gradient Boosting Regression
gbmModel <- train(
  DBH_cm ~ ., 
  data = trainData, 
  method = "gbm",  # Gradient Boosting
  trControl = trainControl,
  tuneLength = 5,  # Number of combinations to try
  verbose = FALSE  # Suppress verbose output
)

# Collect all models into a list
models <- list(
  DecisionTree = dtModel,
  RandomForest = rfModel,
  SVM = svmModel,
  kNN = knnModel,
  GBM = gbmModel
)

# Comparing model performance 
resu2 <- resamples(models)
summary(resu2)

bwplot(resu2, main = "Model Performance Comparison (RMSE)")

# Variable Importance Plots (e.g., Random Forest)
rfImportance <- varImp(rfModel)
plot(rfImportance, main = "Random Forest Variable Importance")

# Make predictions for each model
pred2 <- lapply(models, predict, newdata = testData)

# Evaluate performance metrics (RMSE, R-squared, MAE)
perf <- data.frame(
  Model = names(models),
  RMSE = sapply(pred2, function(pred) sqrt(mean((testData$DBH_cm - pred)^2))),
  Rsquared = sapply(pred2, function(pred) cor(testData$DBH_cm, pred)^2),
  MAE = sapply(pred2, function(pred) mean(abs(testData$DBH_cm - pred)))
)

print(perf)

# Plot observed vs predicted values (e.g., Random Forest)
perfDF <- data.frame(
  Observed = testData$DBH_cm,
  Predicted = pred2$RandomForest
)

ggplot(perfDF, aes(x = Observed, y = Predicted)) +
  geom_point(color = "darkblue", size = 3) +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
  labs(title = "Observed vs Predicted Values (Random Forest)",
       x = "Observed MPG",
       y = "Predicted MPG") +
  theme_minimal()

rm(list = ls(all.names = TRUE)) #Clear all objects 
gc() #Report the memory usage
# end
######################################################################################################################################
