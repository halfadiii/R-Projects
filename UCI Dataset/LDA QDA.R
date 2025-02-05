# Loading required libraries
library(MASS) # library includes LDA and QDA functions
library(ggplot2)
library(dplyr)
library(caret)
library(glmnet)
library(ISLR)

# Seeds data: Loading and preparing the data
seeds_data <- read.table("seeds_dataset.txt", 
                         header=TRUE, col.names=c('Area', 'Perimeter', 'Compactness', 'Kernel_Length',
                                                  'Kernel_Width', 'Asymmetry_Coefficient', 'Kernel_Groove_Length', 'Variety'))

# Selecting features and target for model fitting
features <- seeds_data[c('Kernel_Length', 'Compactness')] 
target <- seeds_data$Variety

# Setting seed for reproducibility
set.seed(42) 
sample_indices <- sample(1:nrow(features), 0.7 * nrow(features)) # Dividing the data into test and train sets
train_features <- features[sample_indices, ]
train_target <- target[sample_indices]
test_features <- features[-sample_indices, ]
test_target <- target[-sample_indices]

# Fitting LDA and QDA
lda_model <- lda(train_target ~ ., data=data.frame(train_features, train_target)) 
lda_pred <- predict(lda_model, test_features)
lda_accuracy <- mean(lda_pred$class == test_target)

qda_model <- qda(train_target ~ ., data=data.frame(train_features, train_target))
qda_pred <- predict(qda_model, test_features)
qda_accuracy <- mean(qda_pred$class == test_target)

print(paste("LDA Accuracy(%):", lda_accuracy * 100)) # Accuracy from LDA
print(paste("QDA Accuracy(%):", qda_accuracy * 100)) # Accuracy from QDA

# Plotting decision boundaries for LDA and QDA
kernel_length_seq <- seq(min(seeds_data$Kernel_Length), max(seeds_data$Kernel_Length), length.out=100)
compactness_seq <- seq(min(seeds_data$Compactness), max(seeds_data$Compactness), length.out=100)
grid <- expand.grid(Kernel_Length=kernel_length_seq, Compactness=compactness_seq)

grid$lda_prediction <- predict(lda_model, grid)$class # LDA predictions
grid$qda_prediction <- predict(qda_model, grid)$class # QDA predictions

lda_plot <- ggplot() + geom_tile(data=grid, aes(x=Kernel_Length, y=Compactness, fill=lda_prediction)) + 
  geom_point(data=seeds_data, aes(x=Kernel_Length, y=Compactness, color=as.factor(Variety)), size=2) +
  scale_fill_manual(values=c("red", "green", "blue")) +
  scale_color_manual(values=c("black", "yellow", "orange")) + ggtitle("LDA Decision Boundary") +
  theme_minimal()

qda_plot <- ggplot() + geom_tile(data=grid, aes(x=Kernel_Length, y=Compactness, fill=qda_prediction)) + 
  geom_point(data=seeds_data, aes(x=Kernel_Length, y=Compactness, color=as.factor(Variety)), size=2) +
  scale_fill_manual(values=c("red", "green", "blue")) +
  scale_color_manual(values=c("black", "yellow", "orange")) + ggtitle("QDA Decision Boundary") +
  theme_minimal()

library(patchwork)
(lda_plot + qda_plot) + plot_layout(ncol=2) # Comparing the graphs