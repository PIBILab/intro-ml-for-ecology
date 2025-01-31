



library(raster)

# Load the TIFF image
image <- raster::brick("AJUBrazil.tif")

plot(image)

# Convert the image to a data frame
pixel_data <- as.data.frame(raster::rasterToPoints(image))

# Rename columns (optional)
names(pixel_data) <- c("x", "y", paste0("band", 1:(ncol(pixel_data) - 2)))

# Set the number of clusters (e.g., 5)
num_clusters <- 5

# Perform k-means clustering
set.seed(123)  # For reproducibility
kmeans_result <- kmeans(pixel_data[, 3:ncol(pixel_data)], centers = num_clusters)

# Add cluster labels to the data frame
pixel_data$cluster <- kmeans_result$cluster

# Create a raster from the clustered data
cluster_raster <- raster::rasterFromXYZ(pixel_data[, c("x", "y", "cluster")])

# Save the clustered image
raster::writeRaster(cluster_raster, "clustered_image.tif", format = "GTiff",overwrite=TRUE)

plot(cluster_raster)

library(rasterVis)

# Plot the clustered image
rasterVis::levelplot(cluster_raster, main = "Clustered Image")



#Semi-Supervised Classification

image <- raster::brick("AJUBrazil.tif")

plot(image)

# Example: Manually labeled pixels
labeled_data <- data.frame(
  x = c(713130, 713026, 714099,714984,712308,713733,714067,712971,715733),  # Example x-coordinates
  y = c(8788055, 8786841, 8785013,8787986,8787806,8788969,8789536,8786467,8789243),    # Example y-coordinates
  class = c("water", "vegetation", "land", "water", "vegetation", "land","water", "vegetation", "land")  # Example labels
)

# Extract pixel values for the labeled coordinates
labeled_data <- cbind(labeled_data, raster::extract(image, labeled_data[, c("x", "y")]))

# Check the structure of your data
str(labeled_data)

# If 'class' is not a factor, convert it
labeled_data$class <- as.factor(labeled_data$class)

library(randomForest)

# Train the model
rf_model <- randomForest(class ~ ., data = labeled_data, ntree = 500, importance = TRUE)

# Prepare the full image data for prediction
full_image_data <- as.data.frame(raster::rasterToPoints(image))

# Predict classes for the entire image
full_image_data$predicted_class <- predict(rf_model, full_image_data)

# Step 1: Create a mapping of class labels to numeric values
class_mapping <- c("water" = 1, "vegetation" = 2, "land" = 3)

# Step 2: Convert the predicted_class column to numeric using the mapping
full_image_data$predicted_class_numeric <- class_mapping[full_image_data$predicted_class]

# Convert the predictions back to a raster
predicted_raster <- raster::rasterFromXYZ(full_image_data[, c("x", "y", "predicted_class_numeric")])

plot(predicted_raster)

# Save the classified image
raster::writeRaster(predicted_raster, "classified_image.tif", format = "GTiff",overwrite=TRUE)


