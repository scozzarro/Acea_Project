# 1.0 Lib ----
library(tidyverse)
library(tidymodels)
library(timetk)
library(modeltime)
library(modeltime.resample)
library(modeltime.ensemble)
library(visdat)

# 2.0 Data ----

data<- read.csv('E:/Data_science/Acea_project/Aquifer_Auser.csv')

summary(data)

vis_dat(data)

vis_miss(data)

# 3.0 Data preparation ----

na_indexes<- which(is.na(data$Depth_to_Groundwater_LT2))

full_data_LT2<- data[-na_indexes,]

full_data_LT2<- full_data_LT2 %>% rename(Date = ï..Date)
full_data_LT2$Date<- as.Date(full_data_LT2$Date, format = c('%d/%m/%Y'))

# 4.0 EDA ----
full_data_LT2 %>% select(Date, Depth_to_Groundwater_LT2) %>%
                  plot_time_series(Date, Depth_to_Groundwater_LT2, .title = 'LT2 Ground Water trend over time')

full_data_LT2 %>% ggplot(aes(Depth_to_Groundwater_LT2)) +
                  geom_boxplot() +
                  coord_flip() +
                  ggtitle('LT2 Ground Water values')

LT2_outliers <- boxplot.stats(full_data_LT2$Depth_to_Groundwater_LT2)$out
LT2_outliers_indexes <- which(full_data_LT2$Depth_to_Groundwater_LT2%in% LT2_outliers)
full_data_LT2<- full_data_LT2[-LT2_outliers_indexes,]

# 5.0 ML Preparation ----
forecast_horizon<- 360

full_data_LT2_tbl<- full_data_LT2 %>% 
                    select(Date, Depth_to_Groundwater_LT2) %>%
                    future_frame(.date_var = Date, .length_out = forecast_horizon, .bind_data = TRUE)
#training Data
train_LT2_tbl<- full_data_LT2_tbl %>% filter(!is.na(Depth_to_Groundwater_LT2))
train_LT2_tbl %>% tk_summary_diagnostics()

#Future Data
future_LT2_tbl<- full_data_LT2_tbl %>% filter(is.na(Depth_to_Groundwater_LT2))
future_LT2_tbl %>% tk_summary_diagnostics()

#time splits
splits<- train_LT2_tbl %>% time_series_split(date_var = Date, assess = forecast_horizon, cumulative = TRUE)
splits

# Preprocessing
recipe_1<- recipe(Depth_to_Groundwater_LT2 ~., training(splits)) %>%
           step_timeseries_signature(Date) %>%
           step_rm(matches('(.iso)|(.xts$)|(hour)|(minute)|(second)|(am.pm)')) %>%
           step_normalize(Date_index.num, Date_year) %>%
           step_mutate(Date_day = factor(Date_day, ordered = TRUE)) %>%
           step_dummy(all_nominal(), one_hot = TRUE)

recipe_1 %>% prep() %>% juice() %>% glimpse()          
recipe_2<- recipe_1 %>% update_role(Date, new_role = 'ID')

recipe_1 %>% prep() %>% summary()
recipe_2 %>% prep() %>% summary()

# 6.0 Models ---- 

# Prophet w/ Regressor
wflw_fit_prophet<- workflow() %>% 
                   add_model(prophet_reg() %>% set_engine('prophet')) %>%
                   add_recipe(recipe_1) %>%
                   fit(training(splits))

# XGBoost
wflw_fit_xgboost<- workflow() %>% 
                   add_model(boost_tree() %>% set_engine('xgboost')) %>%
                   add_recipe(recipe_2) %>%
                   fit(training(splits))

# Random Forest
wflw_fit_rf<- workflow() %>% 
              add_model(rand_forest() %>% set_engine('ranger')) %>%
              add_recipe(recipe_2) %>%
              fit(training(splits))

# SVM 
wflw_fit_svm<- workflow() %>% 
               add_model(svm_rbf() %>% set_engine('kernlab')) %>%
               add_recipe(recipe_2) %>%
               fit(training(splits))

# Prophet Boost
wflw_fit_prophet_boost<- workflow() %>% 
                         add_model(prophet_boost( seasonality_daily = FALSE, 
                                                  seasonality_weekly = FALSE, 
                                                  seasonality_yearly = FALSE) %>% set_engine('prophet_xgboost')) %>%
                         add_recipe(recipe_1) %>%
                         fit(training(splits))

# 7.0 Modeltime workflow ----
submodels_tbl<- modeltime_table(wflw_fit_prophet,
                                wflw_fit_xgboost,
                                wflw_fit_rf,
                                wflw_fit_svm,
                                wflw_fit_prophet_boost)
# Calibrate testing data
submodel_calibrate_tbl<- submodels_tbl %>% modeltime_calibrate(testing(splits))

# Measure Accuracy
submodel_calibrate_tbl %>% modeltime_accuracy()

#Visualize test forecast
submodel_calibrate_tbl %>% modeltime_forecast(new_data = testing(splits),
                                              actual_data = train_LT2_tbl,
                                              keep_data = TRUE) %>%
                           plot_modeltime_forecast()
#Refit

submodels_refit_tbl<- submodel_calibrate_tbl %>%
                      modeltime_refit(train_LT2_tbl)

# Visualize refitted model
submodels_refit_tbl %>% modeltime_forecast(new_data = future_LT2_tbl,
                                           actual_data = train_LT2_tbl,
                                           keep_data = TRUE)%>%
                        plot_modeltime_forecast()

# 8.0 Ensemble models
ensemble_fit_mean<- submodels_tbl %>%
                    filter(!.model_id %in% c(2,4)) %>%
                    ensemble_average(type = 'mean')

ensemble_tbl<- modeltime_table(ensemble_fit_mean)

ensemble_tbl %>% combine_modeltime_tables(submodels_tbl) %>%
                 modeltime_accuracy(testing(splits))

ensemble_tbl %>% modeltime_forecast(new_data = testing(splits),
                                    actual_data = train_LT2_tbl,
                                    keep_data = TRUE) %>%
                plot_modeltime_forecast()

ensemble_refit_tbl<- ensemble_tbl %>% modeltime_refit(train_LT2_tbl)

ensemble_refit_tbl %>% modeltime_forecast(new_data = future_LT2_tbl,
                                          actual_data = train_LT2_tbl,
                                          keep_data = TRUE) %>%
                      plot_modeltime_forecast()
