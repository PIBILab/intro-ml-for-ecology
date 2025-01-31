#' Dr. Eduardo Vinicius da Silva Oliveira
#' PIBILab  - UFS
#' https://github.com/PIBILab/intro-ml-for-ecology
##################################################################
#' Image classification using machine learning algorithms

# -------------------------------------------------------------------------
# Packages

install.packages("raster")  
install.packages("terra")       
install.packages("randomForest")
install.packages("rasterVis")       
library(raster)
library(terra)
library(rasterVis)
library(randomForest)

# -------------------------------------------------------------------------

###
#' K-Means Clustering for Image Classification 
#' Unsupervised learning algorithm

# Load the image
im<-raster::brick("https://github.com/PIBILab/intro-ml-for-ecology/raw/main/2-Image_classification/data/AjuBrazil.tif")

plot(im)

# Convert to a data frame
pix_data <- as.data.frame(raster::rasterToPoints(im))

# Rename columns
names(pix_data) <- c("x", "y", paste0("band", 1:(ncol(pix_data) - 2)))

# Set the number of clusters
num_clusters <- 5

# Perform k-means clustering
set.seed(123)  
kmeans_resu <- kmeans(pix_data[, 3:ncol(pix_data)], centers = num_clusters)

# Add cluster labels to the data frame
pix_data$cluster <- kmeans_resu$cluster

# Create a raster 
clust_raster <- raster::rasterFromXYZ(pix_data[, c("x", "y", "cluster")])

# Save the clustered image
raster::writeRaster(clust_raster, "clust_image.tif", format = "GTiff",overwrite=TRUE)

# Plot the clustered image
plot(clust_raster)
rasterVis::levelplot(clust_raster, main = "Clustered Image")


###
#' Image Classification and Recognition Based Random Forest Algorithm
#' Semi-Supervised Classification

im<-raster::brick("https://github.com/PIBILab/intro-ml-for-ecology/raw/main/2-Image_classification/data/AjuBrazil.tif")

plot(im)

# Manually labeled pixels
lab_data <- data.frame(
  x = c(713130, 713026, 714099,714984,712308,713733,714067,712971,715733),  
  y = c(8788055, 8786841, 8785013,8787986,8787806,8788969,8789536,8786467,8789243),    
  class = c("water", "vegetation", "land", "water", "vegetation", "land","water", "vegetation", "land")  
)

# Extract pixel values for the labeled coordinates
lab_data <- cbind(lab_data, raster::extract(im, lab_data[, c("x", "y")]))

# Convert 'class' in a factor
lab_data$class <- as.factor(lab_data$class)

# Train the model
rf_model <- randomForest(class ~ ., data = lab_data, ntree = 500, importance = TRUE)

# Prepare the full image data for prediction
full_im_data <- as.data.frame(raster::rasterToPoints(im))

# Predict classes for the entire image
full_im_data$predicted_class <- predict(rf_model, full_im_data)

# Create a mapping of class labels to numeric values
class_map <- c("water" = 1, "vegetation" = 2, "land" = 3)

# Convert the predicted_class column to numeric 
full_im_data$predicted_class_num <- class_map[full_im_data$predicted_class]

# Convert the predictions back to a raster
pred_raster <- raster::rasterFromXYZ(full_im_data[, c("x", "y", "predicted_class_num")])

plot(pred_raster)

# Save the classified image
raster::writeRaster(pred_raster, "classified_image.tif", format = "GTiff",overwrite=TRUE)

rm(list = ls(all.names = TRUE)) #Clear all objects 
gc() #Report the memory usage
# end
######################################################################################################################################
