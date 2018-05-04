#' ---
#' title: "3.4: Classifying movie reviews"
#' author: Chollet and Allaire
#' output: github_document
#' ---
#' last update: Thu May 03 15:23:11 2018

library(keras)
library(tidyverse)

imdb <- dataset_imdb(num_words = 10^4)

str(imdb)

# separate the training and testing data

train_data <- imdb$train$x
train_labels <- imdb$train$y

test_data <- imdb$test$x
test_labels <- imdb$test$y

train_data[[1]]
typeof(train_data)

train_labels[[12]]
typeof(train_labels)

max(map_int(train_data, max)) # this should not exceed 10000 since we consider only the top 10000 words

word_index <- dataset_imdb_word_index()
typeof(word_index) # list

word_index$fawn

reverse_word_index <- names(word_index)
names(reverse_word_index) <- word_index
typeof(reverse_word_index)

reverse_word_index[[1]]

decoded_review <- sapply(train_data[[1]], function(index) {
  word <- if (index >= 3) reverse_word_index[[as.character(index - 3)]]
  if (!is.null(word)) word else "?"
})

cat(decoded_review)

vectorize_sequences <- function(sequences, dimension = 10^4) {
  results <- matrix(0, nrow = length(sequences), ncol = dimension) # pre-allocation is good practise
  
  for (i in 1:length(sequences)) {
    results[i, sequences[[i]]] <- 1
  }
  
  return(results)
}

x_train <- vectorize_sequences(train_data)
x_test <- vectorize_sequences(test_data)

str(x_train[1, ])

y_train <- as.numeric(train_labels)
y_test <- as.numeric(test_labels)

#' There are two key decisions one makes in the construction of a multilayer
#' perceptron like the one we are building here. We need to decide how many
#' hidden layers to use and how many neurons per layer.
#'
#' There are broadly three steps - one, build a layered representation of your
#' fancy; two, compile the model (in place) by giving the loss, optimizer and
#' metrics arguments; three, fit the compiled model to data, which returns a
#' history object

keras_model_sequential() %>% 
  layer_dense(units = 16, activation = "relu", input_shape = c(10000)) %>% 
  layer_dense(units = 16, activation = "relu") %>% 
  layer_dense(units = 1, activation = "sigmoid") ->
  model  

compile(model,
        optimizer = optimizer_rmsprop(lr = 0.001),
        loss = loss_binary_crossentropy,
        metrics = metric_binary_accuracy)

val_indices <- 1:10^4

x_val <- x_train[val_indices, ]
partial_x_train <- x_train[-val_indices, ]

y_val <- y_train[val_indices]
partial_y_train <- y_train[-val_indices]

history <- model %>% fit(partial_x_train, partial_y_train, 
                         epochs =  20,
                         batch_size = 512,
                         validation_data = list(x_val, y_val))

str(history)

dev.new()
plot(history) # overfitting is apparent

keras_model_sequential() %>% 
  layer_dense(units = 16, activation = "relu", input_shape = c(10^4)) %>% 
  layer_dense(units = 16, activation = "relu") %>% 
  layer_dense(units = 1, activation = "sigmoid") %>% 
  compile(optimizer = "rmsprop",
          loss = "binary_crossentropy",
          metrics = c("accuracy")) ->
  model

fit(model, x_train, y_train, epochs = 4, batch_size = 512)

results <- evaluate(model, x_test, y_test)

#' Lets vary the number of hidden layers and see the impact on validation 
#' and test accuracy
#' 
#' We begin by dropping one hidden layer

keras_model_sequential() %>% 
  layer_dense(units =  16, activation = "relu", input_shape = c(10^4)) %>% 
  layer_dense(units = 1, activation = "sigmoid") %>% 
  compile(optimizer = "rmsprop",
          loss = "binary_crossentropy",
          metrics = c("accuracy")) ->
  model

fit(model, x_train, y_train, 
    epochs = 4, 
    batch_size = 512, 
    validation_split = 0.2) # validation accuracy did not suffer due to the loss of one hidden layer

results <- evaluate(model, x_test, y_test)
results$acc

#' Let us now add two more hidden layer to the original model, 
#' making it 3 hidden layers of 16 neurons each in total 

keras_model_sequential() %>% 
  layer_dense(units = 16, activation = "relu", input_shape = c(10^4)) %>% 
  layer_dense(units = 16, activation = "relu") %>% 
  layer_dense(units = 16, activation = "relu") %>% 
  layer_dense(units = 16, activation = "relu") %>% 
  layer_dense(units = 1, activation = "sigmoid") %>% 
  compile(optimize = "rmsprop",
          loss = "binary_crossentropy",
          metrics = c("accuracy")) ->
  model

fit(model, x_train, y_train, 
    epochs = 4, 
    batch_size = 512, 
    validation_split = 0.2)

results <- evaluate(model, x_test, y_test)
results$acc # Further addition of layers does not seem to add much value

keras_model_sequential() %>% 
  layer_dense(units = 32, activation = "relu", input_shape = c(10^4)) %>% 
  layer_dense(units = 32, activation = "relu") %>% 
  layer_dense(units = 1, activation = "sigmoid") %>% 
  compile(optimizer = "rmsprop",
          loss = "binary_crossentropy",
          metrics = c("accuracy")) ->
  model

fit(model, x_train, y_train, 
    epochs = 4, 
    batch_size = 512,
    validation_split = 0.2)

#' Change the loss function to MSE
 
keras_model_sequential() %>% 
  layer_dense(units = 32, activation = "relu", input_shape = c(10^4)) %>% 
  layer_dense(units = 32, activation = "relu") %>% 
  layer_dense(units = 1, activation = "sigmoid") %>% 
  compile(optimizer = "rmsprop",
          loss = "mse",
          metrics = c("accuracy")) ->
  model

fit(model, x_train, y_train, 
    epochs = 4, 
    batch_size = 512,
    validation_split = 0.2)

#' Change the activation function to tanh

keras_model_sequential() %>% 
  layer_dense(units = 32, activation = "tanh", input_shape = c(10^4)) %>% 
  layer_dense(units = 32, activation = "tanh") %>% 
  layer_dense(units = 1, activation = "sigmoid") %>% 
  compile(optimizer = "rmsprop",
          loss = "binary_crossentropy",
          metrics = c("accuracy")) ->
  model

fit(model, x_train, y_train, 
    epochs = 4, 
    batch_size = 512,
    validation_split = 0.2)

#' Original results are quite robust to variations in activation functions,
#' number of layers, number of neurons per layer, loss functions, etc