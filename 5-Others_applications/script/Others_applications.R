#' Dr. Eduardo Vinicius da Silva Oliveira
#' PIBILab  - UFS
#' https://github.com/PIBILab/intro-ml-for-ecology
##################################################################
#' Several applications to Machine Learning algorithms

# -------------------------------------------------------------------------
# Packages

install.packages("e1071")
install.packages("caret")
install.packages("randomForest")
install.packages("class")
install.packages("openxlsx")
library(e1071)
library(caret)
library(randomForest)
library(class)
library(openxlsx)

# -------------------------------------------------------------------------

###
#' 1-Soil classification
#' Support vector machines 
 
# Input data
land<-read.xlsx("https://github.com/PIBILab/intro-ml-for-ecology/raw/main/5-Others_applications/data/land_use.xlsx")

land$class <- as.factor(land$class)

# Create partition in the data (80% train, 20% test)
ind <- createDataPartition(land$class, p = 0.8, list = FALSE)
data_train <- land[ind, ]
data_test  <- land[-ind, ]

# Train the model
model_svm <- svm(class ~ band_1 + band_2, data = land)

# Predicting the class
new_data <- data.frame(band_1 = c(160, 165), band_2 = c(115, 125))
pred <- predict(model_svm, new_data)

# Check the results
pred


###
#' 2-Predicting the carbon stocks from vegetation characteristics 
#' Random forest

# Input data
veg<-read.xlsx("https://github.com/PIBILab/intro-ml-for-ecology/raw/main/5-Others_applications/data/vegetation_data.xlsx")

# Train the model
model_rf <- randomForest(carbon_stock ~ wood_density + height, data = veg)

# Making a prediction
new_data <- data.frame(wood_density = c(0.47, 0.52), height = c(10.5, 11))
pred <- predict(model_rf, new_data)

# Check the results
pred

###
#' 3-K-Nearest Neighbors
#' Making predict according neighborhood

# Input data
land<-read.xlsx("https://github.com/PIBILab/intro-ml-for-ecology/raw/main/5-Others_applications/data/land_use.xlsx")

# Train data
new_data <- data.frame(band_1 = c(160, 165), band_2 = c(115, 125))

# Making a prediction
pred <- knn(train = land[, 1:2], test = new_data, cl = land$class, k = 3)
print(pred)

###
#' 4-K-means Clustering 
#' Unsupervised learning
#' Data clustering

# Input data
data<-read.xlsx("https://github.com/PIBILab/intro-ml-for-ecology/raw/main/5-Others_applications/data/climatic_data.xlsx")

# Making K-means for two cluster
model_kmeans <- kmeans(data, centers = 2)

# View the clusters
data$cluster <- as.factor(model_kmeans$cluster)

plot(data$temp, data$prec, col = data$cluster, pch = 16, xlab = "Temperature", ylab = "Precipitation")

rm(list = ls(all.names = TRUE)) #Clear all objects 
gc() #Report the memory usage
# end
########################################################################################################################################
