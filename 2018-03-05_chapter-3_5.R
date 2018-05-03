#' ---
#' title: "Multiclass classification"
#' author: Chollet and Allaire
#' output: github_document
#' ---
#' last update: Thu May 03 16:45:43 2018
#' 
#' Here we classify news into 46 different categories.

library(keras)
library(tidyverse)

reuters <- dataset_reuters(num_words = 10^4)

train_data <- reuters$train$x
train_labels <- reuters$train$y

test_data <- reuters$test$x
test_labels <- reuters$test$y

length(train_data)
length(test_data)

train_data[[1]]

vectorize_sequences <- function(sequences, dimensions = 10^4) {
  
  results <- matrix(0, nrow = length(sequences), ncol = dimensions) # preallocate
  
  for (i in 1:length(sequences)) {
    results[i, sequences[[i]]] <- 1
  }
  
  return(results)
}

x_train <- vectorize_sequences(train_data)
x_test <- vectorize_sequences(test_data)

to_one_hot <- function(labels, dimensions = 46) {
  
  results <- matrix(0, nrow = length(labels), ncol = dimensions)
  
  for(i in 1:length(labels)) {
    results[i, labels[[i]] + 1] <- 1 # notice the + 1
  }
  
  return(results)
}

one_hot_train_labels <- to_one_hot(train_labels)
one_hot_test_labels <- to_one_hot(test_labels)

#' This manual conversion is unnecessary since this is built into keras

one_hot_train_labels <- to_categorical(train_labels)
one_hot_test_labels <- to_categorical(test_labels)

#' In this case the number of labels is 46, so hidden layers with less than
#' 46 neurons will lead to a loss of information

keras_model_sequential() %>% 
  layer_dense(units = 64, activation = "relu", input_shape = c(10^4)) %>% 
  layer_dense(units = 64, activation = "relu") %>% 
  layer_dense(units = 46, activation = "softmax") ->
  model

compile(model,
        optimizer = "rmsprop",
        loss = "categorical_crossentropy",
        metrics = c("accuracy"))

history <- fit(model, x_train, one_hot_train_labels, 
               epochs = 20,
               batch_size = 512,
               validation_split = 0.2)

dev.new()
plot(history) 

#' The above plot shows that overfitting starts to appear after 9 epochs;
#' We rerun the model with 9 epochs

keras_model_sequential() %>% 
  layer_dense(units = 64, activation = "relu", input_shape = c(10^4)) %>% 
  layer_dense(units = 64, activation = "relu") %>% 
  layer_dense(units = 46, activation = "softmax") ->
  model

compile(model,
        optimizer = "rmsprop",
        loss = "categorical_crossentropy",
        metrics = c("accuracy"))

history <- fit(model, x_train, one_hot_train_labels, 
               epochs = 9,
               batch_size = 512,
               validation_split = 0.2)             

plot(history)

predictions <- predict(model, x_test)
dim(predictions)

which.max(predictions[1, ]) # returns the predicted label for the first test data

#' Now lets increase the number of neurons in the layers and see if accuracy
#' improves

keras_model_sequential() %>% 
  layer_dense(units = 128, activation = "relu", input_shape = c(10^4)) %>% 
  layer_dense(units = 128, activation = "relu") %>% 
  layer_dense(units = 46, activation = "softmax") ->
  model

compile(model,
        optimizer = "rmsprop",
        loss = "categorical_crossentropy",
        metrics = c("accuracy"))

history <- fit(model, x_train, one_hot_train_labels, 
               epochs = 9,
               batch_size = 512,
               validation_split = 0.2)

print(history)

#' There seems to be no increase in accuracy. Now let us add two more hidden
#' layers

keras_model_sequential() %>% 
  layer_dense(units = 64, activation = "relu", input_shape = c(10^4)) %>% 
  layer_dense(units = 64, activation = "relu") %>% 
  layer_dense(units = 64, activation = "relu") %>% 
  layer_dense(units = 64, activation = "relu") %>% 
  layer_dense(units = 46, activation = "softmax") ->
  model

compile(model,
        optimizer = "rmsprop",
        loss = "categorical_crossentropy",
        metrics = c("accuracy"))

history <- fit(model, x_train, one_hot_train_labels, 
               epochs = 9,
               batch_size = 512,
               validation_split = 0.2)

print(history)

#' Accuracy actually comes down with increased layers! 
#' 
#' Lets do both increased layers and neurons and see

keras_model_sequential() %>% 
  layer_dense(units = 128, activation = "relu", input_shape = c(10^4)) %>% 
  layer_dense(units = 128, activation = "relu") %>% 
  layer_dense(units = 128, activation = "relu") %>% 
  layer_dense(units = 128, activation = "relu") %>% 
  layer_dense(units = 46, activation = "softmax") ->
  model

compile(model,
        optimizer = "rmsprop",
        loss = "categorical_crossentropy",
        metrics = c("accuracy"))

history <- fit(model, x_train, one_hot_train_labels, 
               epochs = 9,
               batch_size = 512,
               validation_split = 0.2)

print(history) # pretty hopeless
