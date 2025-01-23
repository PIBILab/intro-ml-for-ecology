# =========================================================
# 'MLregress': function to compute decision tree regression in Machine Learning 
# This function uses R libraries 'caret', 'e1071', 'randomForest', 'nnet', 'xgboost', 'klaR', 'C50', and 'ipred'.
# INPUTS: 
#   - 'data': A data.frame containing the data. Columns can be numeric and the target column must be a numerical variable.
#    - 'target': The name of the column in data that contains the response variable. 
# OUTPUTS: 
#   => a list of 2 objects 
#   - 'models': list of models per algorithm 
#   - 'performance': list of statistics per algorithm 
# =========================================================

MLregress <- function(data, target, algorithms = c("rf", "svmLinear", "knn", "lm", "xgbTree", "ridge", "lasso", "earth", "treebag"), seed = 123) {
  
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
  trainIndex <- createDataPartition(data[[target]], p = 0.7, list = FALSE)
  trainData <- data[trainIndex, ]
  testData <- data[-trainIndex, ]
  
  # Cross-validation
  control <- trainControl(method = "cv", number = 5, verboseIter = FALSE)
  
  # Model training
  models <- lapply(algorithms, function(algo) {
    cat("Model training:", algo, "\n")
    tryCatch({
      train(as.formula(paste(target, "~ .")), data = trainData, method = algo, trControl = control)
    }, error = function(e) {
      cat("Error training the model:", algo, "\n", e$message, "\n")
      NULL
    })
  })
  
  names(models) <- algorithms
  
  # Model evaluation
  predictions <- lapply(models, function(model) {
    if (!is.null(model)) {
      tryCatch({
        predict(model, newdata = testData)
      }, error = function(e) {
        cat("Error when predicting for a model:", e$message, "\n")
        NULL
      })
    } else {
      NULL
    }
  })
  
  # Performance metrics
  performance <- lapply(predictions, function(pred) {
    if (!is.null(pred)) {
      tryCatch({
        postResample(pred = pred, obs = testData[[target]])
      }, error = function(e) {
        cat("Error calculating performance metrics:", e$message, "\n")
        NULL
      })
    } else {
      NULL
    }
  })
  
  # Filtering the valid results
  valid_performance <- performance[!sapply(performance, is.null)]
  performance_df <- data.frame(
    Model = names(valid_performance),
    RMSE = sapply(valid_performance, function(x) x["RMSE"]),
    Rsquared = sapply(valid_performance, function(x) x["Rsquared"]),
    MAE = sapply(valid_performance, function(x) x["MAE"])
  )
  
  return(list(models = models, performance = performance_df))
}

# end of function MLregress
########################################################################################################################################