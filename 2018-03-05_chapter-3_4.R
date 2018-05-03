#' ---
#' title: "3.4: Classifying movie reviews"
#' author: Chollet and Allaire
#' output: github_document
#' ---
#' last update: Thu May 03 15:23:11 2018

library(keras)

imdb <- dataset_imdb(num_words = 10^4)

str(imdb)

train_data <- imdb$train$x
train_labels <- imdb$train$y

train_data[[1]]
typeof(train_data)

train_labels[[12]]
typeof(train_labels)
