#' ---
#' title: "Managing workflow with tfruns "
#' author: Pavan Gurazada
#' output: github_document
#' ---
#' last update: Wed May 09 16:16:44 2018

library(keras)
library(tfruns)
library(rsample)
library(recipes)
library(corrr)
library(tidyverse)
library(ggthemes)

theme_set(theme_few())

churn_data_raw <- read_csv("../Econometrics/data/ibm-watson-data/WA_Fn-UseC_-Telco-Customer-Churn.csv", 
                           progress = TRUE)

glimpse(churn_data_raw)

churn_data_raw %>% 
  select(-customerID) %>% 
  drop_na() %>% 
  select(Churn, everything()) ->
  churn_data_tbl

glimpse(churn_data_tbl)

#' Before going ahead, we split the data into training and testing 
train_test_split <- initial_split(churn_data_tbl, prop = 0.8)
train_tbl <- training(train_test_split)
test_tbl <- testing(train_test_split)

glimpse(train_tbl)

recipe(Churn ~ ., data = train_tbl) %>% 
  step_discretize(tenure, options = list(cuts = 6)) %>% # split tenure into 6 bins as discussed earlier
  step_log(TotalCharges, MonthlyCharges) %>% 
  step_dummy(all_nominal(), -all_outcomes()) %>% 
  step_center(all_predictors(), -all_outcomes()) %>% 
  step_scale(all_predictors(), -all_outcomes()) %>% 
  prep(data = train_tbl) ->
  rec_obj

bake(rec_obj, newdata = train_tbl) %>% 
  select(-Churn) ->
  x_train_tbl

glimpse(x_train_tbl)

x_train <- as.matrix(x_train_tbl)

model <- keras_model_sequential() %>% 
  layer_dense(units = 16, 
              activation = 'relu', 
              input_shape = ncol(x_train)) %>% # layer 1
  
  layer_dropout(rate = 0.2) %>% 
  
  layer_dense(units = 16, 
              activation = 'relu') %>% # layer 2
  
  layer_dropout(rate = 0.2) %>% 
  
  layer_dense(units = 16, 
              activation = 'relu') %>% # layer 3
  
  layer_dropout(rate = 0.2) %>% 
  
  layer_dense(units = 16, 
              activation = 'relu') %>% # layer 4
  
  layer_dropout(rate = 0.2) %>% 
  
  layer_dense(units = 1, 
              activation = 'sigmoid')

compile(model, 
        optimizer = optimizer_adam(lr = 0.005),
        loss = 'binary_crossentropy',
        metrics = c('accuracy'))

history <- fit(model, x_train, y_train,
               epochs = 20,
               batch_size = 100, 
               validation_split = 0.2)

k_clear_session()