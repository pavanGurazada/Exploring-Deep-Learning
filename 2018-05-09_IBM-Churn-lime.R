#' ---
#' title: "Explaining the churn model with LIME"
#' author: Pavan Gurazada
#' output: github_document
#' ---

#' last update: Thu May 10 16:26:07 2018

library(keras)
library(lime)
library(rsample)
library(recipes)
library(yardstick)
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

bake(rec_obj, newdata = test_tbl) %>% 
  select(-Churn) ->
  x_test_tbl

x_test <- as.matrix(x_test_tbl)

y_train <- ifelse(pull(train_tbl, Churn) == "Yes", 1, 0)
y_test <- ifelse(pull(test_tbl, Churn) == "Yes", 1, 0)

model_mlp <- keras_model_sequential() %>% 
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

compile(model_mlp, 
        optimizer = optimizer_adam(lr = 0.005),
        loss = 'binary_crossentropy',
        metrics = c('accuracy'))

history <- fit(model_mlp, x_train, y_train,
               epochs = 20,
               batch_size = 100, 
               validation_split = 0.2)

k_clear_session()

yhat_class <- predict_classes(object = model_mlp, 
                              x = x_test) %>% as.vector()

yhat_probs <- predict_proba(object = model_mlp,
                            x = x_test) %>% as.vector()

estimates_test <- data.frame(truth = as.factor(y_test),
                             estimate = as.factor(yhat_class),
                             class_prob = yhat_probs)

#' For most cases the following metrics from the yardstick package are enough
#' to guage the model, especially using the AUC, that accounts for the entire 
#' range of threshold possible given the data

conf_mat(estimates_test, truth, estimate)
metrics(estimates_test, truth , estimate)
roc_auc(estimates_test, truth, class_prob)

#' We now pass the black-box to LIME to explain a set of observations and see which 
#' parameters influence these predictions

class(model_mlp) # Note that this is an S3 class

model_type.keras.models.Sequential <- function(x, ...) {
  'classification'
}

predict_model.keras.models.Sequential <- function(x, newdata, type, ...) {
  pred <- predict_proba(object = x,
                        x = as.matrix(newdata))
  
  return(data.frame(Yes = pred, No = 1 - pred))
}

explainer_mlp <- lime(x = x_train_tbl,
                      model = model_mlp,
                      bin_continuous = FALSE)

explanation <- explain(x_test_tbl[1:10, ],
                       explainer = explainer_mlp,
                       n_labels = 1,
                       n_features = 4,
                       kernel_width = 0.5)

dev.new()

plot_features(explanation) +
  labs(title = 'LIME Feature Importance Visualization') -> p1

plot_explanations(explanation) +
  labs(title = "LIME Feature Importance Visualization") -> p2

ggsave(filename = "lime_test10.png", plot = p1, width = 10, height = 15)
