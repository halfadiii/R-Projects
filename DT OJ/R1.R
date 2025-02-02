# Loading required libraries
library(ISLR)
library(tree)
library(caret)

# Data Preparation
data("OJ")
set.seed(123) # Set the seed for reproducibility
train_indices <- sample(nrow(OJ), 750) # Sampling 750 observations for the training set
train_set <- OJ[train_indices, ] # Training set
test_set <- OJ[-train_indices, ] # Test set

# Building a decision tree model
oj_tree <- tree(Purchase ~ ., data = train_set) # Fit tree with Purchase as the response
summary(oj_tree) # Model summary

# Training error calculation
train_pred <- predict(oj_tree, train_set, type = "class") # Class predictions for the training set
train_error_rate <- mean(train_pred != train_set$Purchase) # Calculation of error rate
print(paste("Training error rate:", train_error_rate * 100, "%")) # Display training error rate

# Tree plot and interpretation
plot(oj_tree) # Plotting the tree
text(oj_tree, cex = 0.8) # Add text to the plot
terminal_nodes <- which(oj_tree$frame$var == "<leaf>") # Finding terminal nodes
num_terminal_nodes <- length(terminal_nodes) # Number of terminal nodes
print(paste("Number of terminal nodes:", num_terminal_nodes)) # Display number of terminal nodes

# Terminal node interpretation
node_index <- 8 # Choosing one of the terminal nodes
node_info <- oj_tree$frame[node_index, ]
if (node_index %in% terminal_nodes) {
  cat(sprintf("Node Index: %d\n", node_index))
  cat(sprintf("Predicted Purchase: %s\n", ifelse(node_info$yval == 1, "CH", "MM")))
  cat(sprintf("Number of Samples in Node: %d\n", node_info$n))
  cat(sprintf("Deviance in Node: %f\n", node_info$dev))
} else {
  cat("Node is not a terminal node. Please verify the node index.\n", node_index)
}

# Test error calculation
test_pred <- predict(oj_tree, test_set, type = "class") # Predicting on the test set
conf_matrix <- confusionMatrix(table(test_pred, test_set$Purchase)) # Creating the confusion matrix
print(conf_matrix) # Display confusion matrix
test_error_rate <- 1 - sum(diag(conf_matrix$table)) / sum(conf_matrix$table) # Calculating the test error rate
print(paste("Test error rate:", test_error_rate * 100, "%")) # Display test error rate

# Cross-validation
cv_oj_tree <- cv.tree(oj_tree, FUN = prune.tree) # Apply cross-validation on the tree
plot(cv_oj_tree$size, cv_oj_tree$dev, type = "b", pch = 19, frame = FALSE, xlab = "Size of the Tree", ylab = "Post Cross-Validation Error Rate", main = "Tree Size vs Error Rate") # Plotting tree size vs error rate

# Optimal tree size from cross-validation
min_error_index <- which.min(cv_oj_tree$dev) # Index of minimum error rate
optimal_tree_size <- cv_oj_tree$size[min_error_index] # Optimal tree size
cat("Tree size corresponding to the lowest cross-validated error rate:", optimal_tree_size)

# Pruning the tree
pruned_tree <- prune.tree(oj_tree, best = optimal_tree_size) # Pruning the tree to optimal size
plot(pruned_tree) # Plotting the pruned tree
text(pruned_tree) # Adding text to the pruned tree plot

# Error comparison between pruned and unpruned trees
unpruned_train_error <- mean(train_pred != train_set$Purchase) # Unpruned tree training error
pruned_train_pred <- predict(pruned_tree, train_set, type = "class") # Predictions from the pruned tree
pruned_train_error <- mean(pruned_train_pred != train_set$Purchase) # Pruned tree training error
cat("Training error rate for the unpruned tree:", unpruned_train_error * 100, "%\n")
cat("Training error rate for the pruned tree:", pruned_train_error * 100, "%\n")

# Test error comparison
unpruned_test_pred <- predict(oj_tree, test_set, type = "class") # Predictions from the unpruned tree
unpruned_test_error <- mean(unpruned_test_pred != test_set$Purchase) # Unpruned tree test error
pruned_test_pred <- predict(pruned_tree, test_set, type = "class") # Predictions from the pruned tree
pruned_test_error <- mean(pruned_test_pred != test_set$Purchase) # Pruned tree test error
cat("Test error rate for the unpruned tree:", unpruned_test_error * 100, "%\n")
cat("Test error rate for the pruned tree:", pruned_test_error * 100, "%\n")
