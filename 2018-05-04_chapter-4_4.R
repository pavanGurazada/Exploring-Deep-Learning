#' ---
#' title: "4.4 Fighting overfitting"
#' author: Chollet and Allaire
#' output: github_document
#' ---
#' last update: Fri May 04 06:28:24 2018

library(keras)
library(tidyverse)

imdb <- dataset_imdb(num_words = 10000)

train_data <- imdb$train$x
train_labels <- imdb$train$y

test_data <- imdb$test$x
test_labels <- imdb$test$y

vectorize_sequences <- function(sequences, dimension = 10^4) {
  # Create an all-zero matrix of shape (len(sequences), dimension)
  results <- matrix(0, nrow = length(sequences), ncol = dimension)
  
  for (i in 1:length(sequences))
    # Sets specific indices of results[i] to 1s
    results[i, sequences[[i]]] <- 1
  
  return(results)
}

# Our vectorized training data
x_train <- vectorize_sequences(train_data)
# Our vectorized test data
x_test <- vectorize_sequences(test_data)
# Our vectorized labels
y_train <- as.numeric(train_labels)
y_test <- as.numeric(test_labels)

#' A good way to fight overfitting is to reduce the network size.

original_model <- keras_model_sequential() %>% 
  layer_dense(units = 16, activation = "relu", input_shape = c(10^4)) %>% 
  layer_dense(units = 16, activation = "relu") %>% 
  layer_dense(units = 1, activation = "sigmoid")

compile(original_model,
        optimizer = "rmsprop",
        loss = "binary_crossentropy",
        metrics = c("accuracy"))

smaller_model <- keras_model_sequential() %>% 
  layer_dense(units = 4, activation = "relu", input_shape = c(10^4)) %>% 
  layer_dense(units = 4, activation = "relu") %>% 
  layer_dense(units = 1, activation = "sigmoid")

compile(smaller_model,
        optimizer = "rmsprop",
        loss = "binary_crossentropy",
        metrics = c("accuracy"))


original_history <- fit(original_model, x_train, y_train, 
                        epochs = 20, 
                        batch_size = 512, 
                        validation_data = list(x_test, y_test))

smaller_history <- fit(smaller_model, x_train, y_train,
                       epochs = 20,
                       batch_size = 512,
                       validation_data = list(x_test, y_test))

plot_training_losses <- function(losses) {
  loss_names <- names(losses)
  losses <- as.data.frame(losses)
  losses$epoch <- seq_len(nrow(losses))
  
  losses %>% 
    gather(model, loss, loss_names[[1]], loss_names[[2]]) %>% 
    ggplot(aes(x = epoch, y = loss, color = model)) +
    geom_line()
}

dev.new()
plot_training_losses(list(original_model = original_history$metrics$val_loss,
                          smaller_model = smaller_history$metrics$val_loss))

bigger_model <- keras_model_sequential() %>% 
  layer_dense(units = 512, activation = "relu", input_shape = c(10^4)) %>% 
  layer_dense(units = 512, activation = "relu") %>% 
  layer_dense(units = 1, activation = "sigmoid")

compile(bigger_model,
        optimizer = "rmsprop",
        loss = "binary_crossentropy",
        metrics = c("accuracy"))

bigger_history <- fit(bigger_model, x_train, y_train,
                      epochs = 20,
                      batch_size = 512,
                      validation_data = list(x_test, y_test))

plot_training_losses(list(original_model = original_history$metrics$val_loss,
                          bigger_model = bigger_history$metrics$val_loss))

#' Another way to combat overfitting is to add regularization so that the
#' weights are controlled

l2_model <- keras_model_sequential() %>% 
  layer_dense(units = 16, kernel_regularizer = regularizer_l2(0.001),
              activation = "relu", input_shape = c(10^4)) %>%
  layer_dense(units = 16, kernel_regularizer = regularizer_l2(0.001),
              activation = "relu") %>% 
  layer_dense(units = 1, activation = "sigmoid")

compile(l2_model, 
        optimizer = "rmsprop",
        loss = "binary_crossentropy",
        metrics = c("accuracy"))  

l2_model_history <- fit(l2_model, x_train, y_train, 
                        epochs = 20, 
                        batch_size = 512,
                        validation_data = list(x_test, y_test))

plot_training_losses(list(original_model = original_history$metrics$val_loss,
                          l2_model = l2_model_history$metrics$val_loss))

#' Another way to reduce overfitting is to drop a portion of information
#' assembles in the neurons composed in the hidden layer

keras_model_sequential() %>% 
  layer_dense(units = 16, activation = "relu", input_shape = c(10^4)) %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 16, activation = "relu") %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 1, activation = "sigmoid") ->
  dpt_model

compile(dpt_model, 
        optimizer = "rmsprop",
        loss = "binary_crossentropy",
        metrics = c("accuracy"))

dpt_history <- fit(dpt_model, x_train, y_train,
                   epochs = 20,
                   batch_size = 512,
                   validation_data = list(x_test, y_test))

#' So which of these overfitting reducers better?

plot_training_losses(list(l2_model = l2_model_history$metrics$val_loss,
                          dpt_model = dpt_history$metrics$val_loss))

#' The above plot suggests that regularization is better