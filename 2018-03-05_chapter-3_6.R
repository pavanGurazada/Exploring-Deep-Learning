#' ---
#' title: "3.6: Regression on housing prices"
#' author: Chollet and Allaire
#' output: github_document
#' ---
#' last update: Thu May 03 19:51:09 2018

library(keras)
library(tidyverse)

boston <- dataset_boston_housing()

train_data <- boston$train$x
train_target <- boston$train$y

test_data <- boston$test$x
test_target <- boston$test$y

str(train_data)
str(train_target)

#' Standardizing the features is a standard procedure prior to network building

train_data <- scale(train_data)
test_data <- scale(test_data)

#' Our plan is to use K-fold cross validation so we need to build the network 
#' multiple times

build_model <- function() {
  model <- keras_model_sequential() %>% 
    layer_dense(units = 64, activation = "relu", 
                input_shape = dim(train_data)[[2]]) %>% 
    layer_dense(units = 64, activation = "relu") %>% 
    layer_dense(units = 1) 
  
  model %>% compile(
    optimizer = "rmsprop", 
    loss = "mse", 
    metrics = c("mae")
  )
}

k <- 4
indices <- sample(1:nrow(train_data))
folds <- cut(1:length(indices), breaks = k, labels = FALSE) 
num_epochs <- 100
all_scores <- c()
for (i in 1:k) {
  cat("processing fold #", i, "\n")
  # Prepare the validation data: data from partition # k
  val_indices <- which(folds == i, arr.ind = TRUE) 
  val_data <- train_data[val_indices,]
  val_targets <- train_targets[val_indices]
  
  # Prepare the training data: data from all other partitions
  partial_train_data <- train_data[-val_indices,]
  partial_train_targets <- train_targets[-val_indices]
  
  # Build the Keras model (already compiled)
  model <- build_model()
  
  # Train the model (in silent mode, verbose=0)
  model %>% fit(partial_train_data, partial_train_targets,
                epochs = num_epochs, batch_size = 1, verbose = 0)
  
  # Evaluate the model on the validation data
  results <- model %>% evaluate(val_data, val_targets, verbose = 0)
  all_scores <- c(all_scores, results$mean_absolute_error)
}  

mean(all_scores)

k_clear_session() # reset all models from before

num_epochs <- 500

all_mae_histories <- NULL

for (i in 1:k) {
  cat("processing fold", i, "at: ", format(Sys.time(), "%H:%M"), "\n")
  
  # Prepare the validation data: data from partition # k
  val_indices <- which(folds == i, arr.ind = TRUE)
  val_data <- train_data[val_indices,]
  val_targets <- train_targets[val_indices]
  
  # Prepare the training data: data from all other partitions
  partial_train_data <- train_data[-val_indices,]
  partial_train_targets <- train_targets[-val_indices]
  
  # Build the Keras model (already compiled)
  model <- build_model()
  
  # Train the model (in silent mode, verbose=0)
  history <- fit(model, partial_train_data, partial_train_targets,
                 validation_data = list(val_data, val_targets),
                 epochs = num_epochs, batch_size = 1, verbose = 0)
  
  mae_history <- history$metrics$val_mean_absolute_error
  all_mae_histories <- rbind(all_mae_histories, mae_history)  
}

average_mae_history <- data.frame(epoch = seq(1:ncol(all_mae_histories)),
                                  validation_mae = apply(all_mae_histories, 2, mean))

dev.new()
ggplot(average_mae_history) +
  geom_line(aes(x = epoch, y = validation_mae))

ggplot(average_mae_history) +
  geom_smooth(aes(x = epoch, y = validation_mae))
