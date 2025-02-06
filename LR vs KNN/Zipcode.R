# Storing the paths of the train and the test data
train_path <-"C:\Adi\GitHub\R-Projects\LR vs KNN\zip_train.RData"
test_path <- "C:\Adi\GitHub\R-Projects\LR vs KNN\zip_test.RData"

# Loading the data
loaded_train <- load(train_path)
loaded_test <- load(test_path)

# Assigning the data
train_data <- get(loaded_train[1])
test_data <- get(loaded_test[1])

# Filter for digits 1 and 7
train_data <- train_data[train_data[, 1] %in% c(1, 7), ]
test_data <- test_data[test_data[, 1] %in% c(1, 7), ]

# Get the digits from the first column of both datasets
train_labels <- train_data[, 1]
test_labels <- test_data[, 1]

#linear Regression
lin_model <- lm(train_labels ~ ., data =as.data.frame(train_data))# Fit the linear model
train_pred_lin <- predict(lin_model, newdata =as.data.frame(train_data)) # Making predictions
test_pred_lin <- predict(lin_model, newdata = as.data.frame(test_data))
train_error_lin <-mean((train_labels - train_pred_lin)^2) # Calculating the mean squared error
test_error_lin <-mean((test_labels - test_pred_lin)^2)#Calculating the mean squared error

# KNN
library(class)
k_values <- seq(1, 17, by = 2) # Creating odd numbers between 1 to 17 as given
train_errors_knn <- numeric(length(k_values)) # Store the errors
test_errors_knn <- numeric(length(k_values))#Store the errors

for (i in seq_along(k_values)) {
  k <-k_values[i]
  knn_pred_train <- knn(train = train_data[, -1], test = train_data[, -1], cl = train_labels, k = k) # Give the true labels
  knn_pred_test <- knn(train = train_data[, -1], test = test_data[, -1], cl = train_labels, k = k)
  train_errors_knn[i] <- mean(train_labels != knn_pred_train)
  test_errors_knn[i] <- mean(test_labels != knn_pred_test)
}

#plotting the graph
plot(k_values, train_errors_knn, type = 'b', col = 'blue', pch = 19, xlab = "k values", ylab = "Error Rate",ylim=range(c(train_errors_knn, test_errors_knn)),main="KNN Training and Test Errors")
lines(k_values, test_errors_knn, type = 'b', col = 'red', pch=18)
points(k_values, test_errors_knn, type = 'b', col = 'red', pch=18)
legend("topright", inset = c(-0.3, 0), legend = c("Training Error","Test Error"), col = c("blue", "red"), pch = c(19, 18), xpd = TRUE)
