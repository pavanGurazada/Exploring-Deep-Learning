#' ---
#' title: "Revisiting the IBM churn case "
#' author: Pavan Gurazada
#' output: github_document
#' ---

library(keras)
library(onehot)
library(rsample)
library(recipes)
library(corrr)
library(yardstick)
library(tidyverse)
library(ggthemes)
library(caret)

theme_set(theme_few())

churn_data_raw <- read_csv("../Econometrics/data/ibm-watson-data/WA_Fn-UseC_-Telco-Customer-Churn.csv", 
                           progress = TRUE)

glimpse(churn_data_raw)

summary(churn_data_raw) # clean data with missing values only in the total charges column

#' We take out the customer id since it is redundant and reorder the data frame
#' by pushing the dependent variable to the front. We take out missing values as
#' well.

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

#' There are several categorical variables which will need one-hot encoding.
#' Lets look at the distribution of the numeric variables - tenure, monthly
#' charges and total charges

dev.new()

ggplot(train_tbl) +
  geom_histogram(aes(x = tenure), fill = "black", color = "white", bins = max(train_tbl$tenure)) +
  labs(x = "Tenure (months)",
       title = "Distribution of tenure")

#' the above plot shows that there are quite a few people on either ends of the
#' spectrum - those who have been with the company for long and those who have
#' not. It might be beneficial to divide these into a set of categories split by
#' the number of years the customers have been with the firm. The logic is that
#' cohorts of customers who have been with the company together might be
#' behaving in a similar way. In general, this is a good approach for count
#' variables amongst the predictors. We can tweak the number of bins to ensure
#' roughly the same number of observations fall in each bin

ggplot(train_tbl) +
  geom_histogram(aes(x = tenure), fill = "black", color = "white", bins = 6) +
  labs(x = "Tenure (months)",
       title = "Distribution of tenure")

#' The above plot normalizes the counts well. This can be tweaked by manually
#' changing the number of bins, i.e., categories into which the data can be
#' grouped; below we explore binning several categories

b <- 5 # change this value between 4 and 10 and plot the histogram

ggplot(train_tbl) +
  geom_histogram(aes(x = tenure), 
                 bins = b,
                 fill = "black",
                 color = "white") +
  labs(x = "Tenure (months)")


#' Now moving on to the monthly charges

ggplot(train_tbl) +
  geom_histogram(aes(x = MonthlyCharges), 
                 bins = 50, 
                 fill = "black", 
                 color = "white")

#' The data shows a skew towards the left. Lets see if a log transformation  helps
#' with the skew

ggplot(train_tbl) +
  geom_histogram(aes(x = log(MonthlyCharges)), 
                 bins = 50, 
                 fill = "black", 
                 color = "white")

#' The above plot shows that the data does indeed get spaced out better. We can
#' formalize if we can gain any improvements by a log transform by checking if
#' the correlation between the predictor and outcome increases by performing the
#' transform

train_tbl %>% 
  select(Churn, MonthlyCharges) %>% 
  mutate(Churn = Churn %>% as.factor() %>% as.numeric(),
         LogMonthlyCharges = log(MonthlyCharges)) %>% 
  correlate() %>% 
  focus(Churn) 

#' A log transformation helped in this case, so we power ahead to the Total
#' charges.

ggplot(train_tbl) +
  geom_histogram(aes(x = TotalCharges),
                 bins = 100,
                 fill = "black",
                 color = "white")

#' This feature also shows a high skew.  Lets again apply a log transformation

ggplot(train_tbl) +
  geom_histogram(aes(x = log(TotalCharges)),
                 bins = 100,
                 fill = "black",
                 color = "white")

#' Lets make the impact of the log transformation formal

train_tbl %>% 
  select(Churn, TotalCharges) %>% 
  mutate(Churn = Churn %>% as.factor() %>% as.numeric(),
         LogTotalCharges = log(TotalCharges)) %>% 
  correlate() %>% 
  focus(Churn)

#' Once again, a log transformation improves the correlation; we now put together
#' all the exploration here into a recipe for pre-processing.
#' 
#' The preprocessing steps are collected into a recipe object that is then 
#' baked with the data as the input

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

#' Now we need to use the same pre-processing parameters of the training set to 
#' transform the test set. This is important so that data does not leak

bake(rec_obj, newdata = test_tbl) %>% 
  select(-Churn) ->
  x_test_tbl

y_train <- ifelse(pull(train_tbl, Churn) == "Yes", 1, 0)
y_test <- ifelse(pull(test_tbl, Churn) == "Yes", 1, 0)

#' *Building the MLP*

model_2h <- keras_model_sequential() %>% 
  layer_dense(units = 16, 
              activation = 'relu', 
              input_shape = ncol(x_train)) %>%
  
  layer_dropout(rate = 0.2) %>% 
  
  layer_dense(units = 16, 
              activation = 'relu') %>%
  
  layer_dropout(rate = 0.2) %>% 
  
  layer_dense(units = 1, 
              activation = 'sigmoid')

compile(model_2h, 
        optimizer = 'adam',
        loss = 'binary_crossentropy',
        metrics = c('accuracy'))

history_2h <- fit(model_2h, x_train, y_train,
                  epochs = 20,
                  batch_size = 100, 
                  validation_split = 0.2)

k_clear_session()

model_4h <- keras_model_sequential() %>% 
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

compile(model_4h, 
        optimizer = 'adam',
        loss = 'binary_crossentropy',
        metrics = c('accuracy'))

history_4h <- fit(model_4h, x_train, y_train,
                  epochs = 20,
                  batch_size = 100, 
                  validation_split = 0.2)

#' Four hidden layers increases the accuracy by a little bit. Lets try increasing
#' the number of hidden layers to 8

k_clear_session()

model_8h <- keras_model_sequential() %>% 
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
  
  layer_dense(units = 16, 
              activation = 'relu') %>% # layer 5
  
  layer_dropout(rate = 0.2) %>% 
  
  layer_dense(units = 16, 
              activation = 'relu') %>% # layer 6
  
  layer_dropout(rate = 0.2) %>% 

  layer_dense(units = 16, 
              activation = 'relu') %>% # layer 7
  
  layer_dropout(rate = 0.2) %>% 
  
  layer_dense(units = 16, 
              activation = 'relu') %>% # layer 8
  
  layer_dropout(rate = 0.2) %>% 
  
  layer_dense(units = 1, 
              activation = 'sigmoid')

compile(model_8h, 
        optimizer = 'adam',
        loss = 'binary_crossentropy',
        metrics = c('accuracy'))

history_8h <- fit(model_8h, x_train, y_train,
                  epochs = 20,
                  batch_size = 100, 
                  validation_split = 0.2)

k_clear_session()

model_24u <- keras_model_sequential() %>% 
  layer_dense(units = 24, 
              activation = 'relu', 
              input_shape = ncol(x_train)) %>% # layer 1
  
  layer_dropout(rate = 0.2) %>% 
  
  layer_dense(units = 24, 
              activation = 'relu') %>% # layer 2
  
  layer_dropout(rate = 0.2) %>% 
  
  layer_dense(units = 24, 
              activation = 'relu') %>% # layer 3
  
  layer_dropout(rate = 0.2) %>% 
  
  layer_dense(units = 24, 
              activation = 'relu') %>% # layer 4
  
  layer_dropout(rate = 0.2) %>% 
  
  layer_dense(units = 1, 
              activation = 'sigmoid')

compile(model_24u, 
        optimizer = 'adam',
        loss = 'binary_crossentropy',
        metrics = c('accuracy'))

history_24u <- fit(model_24u, x_train, y_train,
                   epochs = 10,
                   batch_size = 100, 
                   validation_split = 0.2)

print(history_24u)

#' Looks like 4 hidden layers with 16 units per layer perfors the best
#' We refit the model to the entire training data at these parameters

k_clear_session()

model_4h <- keras_model_sequential() %>% 
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

compile(model_4h, 
        optimizer = 'adam',
        loss = 'binary_crossentropy',
        metrics = c('accuracy'))

history_4h <- fit(model_4h, x_train, y_train,
                  epochs = 10,
                  batch_size = 100)

y_hat_class <- predict_classes(model_4h, x = as.matrix(x_test_tbl)) %>% as.vector()
y_hat_probs <- predict_proba(model_4h, x = as.matrix(x_test_tbl)) %>% as.vector()

estimates_mlp <- data.frame(truth = as.factor(y_test) %>% fct_recode(yes = "1", no = "0"),
                            estimate = as.factor(y_hat_class) %>% fct_recode(yes = "1", no = "0"),
                            class_prob = y_hat_probs)

conf_mat(estimates_mlp, truth, estimate)
roc_auc(estimates_mlp, truth, class_prob)

#' *Additional analysis*
#' One variable we engineered before starting the analysis is to divide tenure
#' into 6 bins. One point to ponder is whether this had a significant impact on
#' what the neural network could learn. Lets see if we instead increased this to
#' the actual number of unique values of tenure in the data. Does this increase
#' the performance of the neural network?
#' 
#' To do this we manually one-hot encode all the features.

glimpse(train_tbl)

train_tbl %>% select(-Churn, -MonthlyCharges, -TotalCharges) %>% 
              mutate_all(as.factor) -> 
              train_fct

encoder <- onehot(train_fct, max_levels = 75)
train_fct <- predict(encoder, train_fct)

train_df <- cbind(train_fct, select(train_tbl, Churn, MonthlyCharges, TotalCharges)) %>% 
                select(Churn, everything()) %>% 
                mutate(Churn = ifelse(Churn == "Yes", 1, 0))

train_df %>% 
  select(-Churn) %>% 
  scale() %>% 
  as.matrix() ->
  x_train

y_train <- ifelse(pull(train_tbl, Churn) == "Yes", 1, 0)
y_test <- ifelse(pull(test_tbl, Churn) == "Yes", 1, 0)

#' Let us fit the best model from before to this data with higher dimensions

k_clear_session()

model_10h <- keras_model_sequential() %>% 
  layer_dense(units = 24, 
              activation = 'relu', 
              input_shape = ncol(x_train)) %>% # layer 1
  
  layer_dropout(rate = 0.2) %>% 
  
  layer_dense(units = 24, 
              activation = 'relu') %>% # layer 2
  
  layer_dropout(rate = 0.2) %>% 
  
  layer_dense(units = 24, 
              activation = 'relu') %>% # layer 3
  
  layer_dropout(rate = 0.2) %>% 
  
  layer_dense(units = 24, 
              activation = 'relu') %>% # layer 4
  
  layer_dropout(rate = 0.2) %>% 
  
  layer_dense(units = 24, 
              activation = 'relu') %>% # layer 5
  
  layer_dropout(rate = 0.2) %>% 
  
  layer_dense(units = 24, 
              activation = 'relu') %>% # layer 6
  
  layer_dropout(rate = 0.2) %>% 
  
  layer_dense(units = 24, 
              activation = 'relu') %>% # layer 7
  
  layer_dropout(rate = 0.2) %>% 
  
  layer_dense(units = 24, 
              activation = 'relu') %>% # layer 8
  
  layer_dropout(rate = 0.2) %>% 
  
  layer_dense(units = 24, 
              activation = 'relu') %>% # layer 9
  
  layer_dropout(rate = 0.2) %>% 
  
  layer_dense(units = 24, 
              activation = 'relu') %>% # layer 10
  
  layer_dropout(rate = 0.2) %>% 
  
  layer_dense(units = 1, 
              activation = 'sigmoid')

compile(model_10h, 
        optimizer = 'adam',
        loss = 'binary_crossentropy',
        metrics = c('accuracy'))

history_10h <- fit(model_10h, x_train, y_train,
                  epochs = 20,
                  batch_size = 10,
                  validation_split = 0.2)

