#' ---
# Title: R Codes for an Introduction to Machine Learning in Ecology 
# https://github.com/PIBILab/intro-ml-for-ecology
# by Dr. Eduardo V. S. Oliveira
# 23/01/2025
#' ---

# Compute decision tree in Machine Learning---------

#####Tree classification#####

### Libraries

if(!require(caret)) install.packages("caret")
if(!require(e1071)) install.packages("e1071")
if(!require(randomForest)) install.packages("randomForest")
if(!require(nnet)) install.packages("nnet")
if(!require(xgboost)) install.packages("xgboost")
if(!require(klaR)) install.packages("klaR")
if(!require(C50)) install.packages("C50")
if(!require(ipred)) install.packages("ipred")
if(!require(openxlsx)) install.packages("openxlsx")

### Load function

source("https://github.com/PIBILab/intro-ml-for-ecology/raw/main/1-Decision_trees/functions/MLclass.R")

### Input data 

data1<-read.xlsx("https://github.com/PIBILab/intro-ml-for-ecology/raw/main/1-Decision_trees/data/data_urb_affor.xlsx")

### Running the function

resu1<-MLclass(data1, target = "state")

#View results
resu1$accuracy

## Making the prediction for a new data set

# New data for predictions

newdata<-read.xlsx("https://github.com/PIBILab/intro-ml-for-ecology/raw/main/1-Decision_trees/data/pred_urb_affor.xlsx")

# Select the model adjusted
treebag_model <- resu1$models[["treebag"]]

foreTC <- predict(treebag_model, newdata = newdata)

# Probabilities of classes
prob <- predict(treebag_model, newdata = newdata, type = "prob")

head(prob)


#####Tree regression#####

packs <- c("caret", "e1071", "randomForest", "nnet", "xgboost", "klaR", "C50", "ipred","elasticnet")
install.packages(setdiff(packs, installed.packages()[,"Package"]))

### Load function

source("https://github.com/PIBILab/intro-ml-for-ecology/raw/main/1-Decision_trees/functions/MLregress.R")

### Input data 

data2<-read.xlsx("https://github.com/PIBILab/intro-ml-for-ecology/raw/main/1-Decision_trees/data/biomass_urb_affor.xlsx")

### Running the function

resu2<-MLregress(data2, target = "biomass", algorithms = c("rf", "svmLinear", "knn", "lm", "xgbTree", "ridge", "lasso", "earth", "treebag"))

#View results
resu2$performance

## Making the prediction for a new data set

# New data for predictions

newdata<-read.xlsx("https://github.com/PIBILab/intro-ml-for-ecology/raw/main/1-Decision_trees/data/pred_urb_affor.xlsx")

# Select the model adjusted
svm_model <- resu2$models[["svmLinear"]]

foreTR <- predict(svm_model, newdata = newdata)

rm(list = ls(all.names = TRUE)) #Clear all objects 
gc() #Report the memory usage
# end
########################################################################################################################################
