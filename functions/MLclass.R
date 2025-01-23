# =========================================================
# 'MLclass': function to compute decision tree classification in Machine Learning 
# This function uses R libraries 'caret', 'e1071', 'randomForest', 'nnet', 'xgboost', 'klaR', 'C50', and 'ipred'.
# INPUTS: 
#   - 'data': A data.frame containing the data. Columns can be numeric or categorical and the target column must be a categorical variable.
#    - 'target': The name of the column in data that contains the response variable. 
# OUTPUTS: 
#   => a list of 3 objects 
#   - 'models': list of models per algorithm 
#   - 'confusion_matrices': list of confusion matrices and statistics per algorithm 
#   - 'accuracy': a data.frame
# =========================================================

MLclass <- function(data, target, algorithms = c("rf", "svmLinear", "knn", "glm", "nnet", "xgbTree","nb", "C5.0", "treebag"),seed = 123) {
  set.seed(seed)
  
  # libraries required 
  require(caret)
  require(e1071)
  require(randomForest)
  require(nnet)
  require(xgboost)
  require(klaR)
  require(C50)
  require(ipred)
  
  # Data preparation
  data[[target]] <- as.factor(data[[target]])
  trainIndex <- createDataPartition(data[[target]], p = 0.7, list = FALSE)
  trainData <- data[trainIndex, ]
  testData <- data[-trainIndex, ]
  
  # Cross-validation
  control <- trainControl(method = "cv", number = 5, verboseIter = FALSE)
  
  # Model training
  models <- lapply(algorithms, function(algo) {
    cat("Model training:", algo, "\n")
    train(as.formula(paste(target, "~ .")), data = trainData, method = algo, trControl = control)
  })
  
  names(models) <- algorithms
  
  # Model evaluation
  forecast <- lapply(models, predict, newdata = testData)
  confusionMatrices <- lapply(forecast, function(pred) {
    confusionMatrix(pred, testData[[target]])
  })
  
  # Accuracy metrics
  accuracy <- sapply(confusionMatrices, function(cm) cm$overall["Accuracy"])
  results <- data.frame(Model = names(accuracy), Accuracy = accuracy)
  
  return(list(models = models, confusion_matrices = confusionMatrices, accuracy = results))
}


# end of function MLclass
########################################################################################################################################
