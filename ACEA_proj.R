# 1.0 Lib ----
library(tidyverse)
library(tidymodels)
library(timetk)
library(modeltime)
library(modeltime.resample)
library(modeltime.ensemble)
library(visdat)
library(corrplot)


# 2.0 Data ----

data<- read.csv('E:/Data_science/Acea_project/Aquifer_Auser.csv')

summary(data)

vis_dat(data)

vis_miss(data)

data<- data %>% rename(Date = ï..Date)
# 3.0 Data preparation ----

na_indexes<- which(is.na(data$Depth_to_Groundwater_LT2))
na_indexes_SAL<- which(is.na(data$Depth_to_Groundwater_SAL))


full_data_LT2<- data[-na_indexes,]
full_data_SAL<- data[-na_indexes_SAL,]


full_data_LT2$Date<- as.Date(full_data_LT2$Date, format = c('%d/%m/%Y'))

full_data_SAL$Date<- as.Date(full_data_SAL$Date, format = c('%d/%m/%Y'))


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

full_data_SAL %>% select(Date, Depth_to_Groundwater_SAL) %>%
                  plot_time_series(Date, Depth_to_Groundwater_SAL, .title = 'SAL Ground Water trend over time')

full_data_SAL %>% ggplot(aes(Depth_to_Groundwater_SAL)) +
                  geom_boxplot() +
                  coord_flip() +
                  ggtitle('SAL Ground Water values')

SAL_outliers <- boxplot.stats(full_data_SAL$Depth_to_Groundwater_SAL)$out
SAL_outliers_indexes <- which(full_data_SAL$Depth_to_Groundwater_SAL %in% SAL_outliers)
full_data_SAL<- full_data_SAL[-SAL_outliers_indexes,]

# 5.0 ML Preparation ----
forecast_horizon<- 360

full_data_LT2_tbl<- full_data_LT2 %>% 
                    select(Date, Depth_to_Groundwater_LT2) %>%
                    future_frame(.date_var = Date, .length_out = forecast_horizon, .bind_data = TRUE)

full_data_SAL_tbl<- full_data_SAL %>% 
                    select(Date, Depth_to_Groundwater_SAL) %>%
                    future_frame(.date_var = Date, .length_out = forecast_horizon, .bind_data = TRUE)

#training Data
train_LT2_tbl<- full_data_LT2_tbl %>% filter(!is.na(Depth_to_Groundwater_LT2))
train_LT2_tbl %>% tk_summary_diagnostics()

train_SAL_tbl<- full_data_SAL_tbl %>% filter(!is.na(Depth_to_Groundwater_SAL))
train_SAL_tbl %>% tk_summary_diagnostics()

#Future Data
future_LT2_tbl<- full_data_LT2_tbl %>% filter(is.na(Depth_to_Groundwater_LT2))
future_LT2_tbl %>% tk_summary_diagnostics()

future_SAL_tbl<- full_data_SAL_tbl %>% filter(is.na(Depth_to_Groundwater_SAL))
future_SAL_tbl %>% tk_summary_diagnostics()

#time splits
splits<- train_LT2_tbl %>% time_series_split(date_var = Date, assess = forecast_horizon, cumulative = TRUE)
splits

splits_SAL<- train_SAL_tbl %>% time_series_split(date_var = Date, assess = forecast_horizon, cumulative = TRUE)
splits_SAL

# Preprocessing
recipe_1<- recipe(Depth_to_Groundwater_LT2 ~., training(splits)) %>%
           step_timeseries_signature(Date) %>%
           step_rm(matches('(.iso)|(.xts$)|(hour)|(minute)|(second)|(am.pm)')) %>%
           step_normalize(Date_index.num, Date_year) %>%
           step_mutate(Date_day = factor(Date_day, ordered = TRUE)) %>%
           step_dummy(all_nominal(), one_hot = TRUE)

recipe_1_SAL<- recipe(Depth_to_Groundwater_SAL ~., training(splits_SAL)) %>%
               step_timeseries_signature(Date) %>%
               step_rm(matches('(.iso)|(.xts$)|(hour)|(minute)|(second)|(am.pm)')) %>%
               step_normalize(Date_index.num, Date_year) %>%
               step_mutate(Date_day = factor(Date_day, ordered = TRUE)) %>%
               step_dummy(all_nominal(), one_hot = TRUE)



recipe_1 %>% prep() %>% juice() %>% glimpse()
recipe_1_SAL %>% prep() %>% juice() %>% glimpse()

recipe_2<- recipe_1 %>% update_role(Date, new_role = 'ID')
recipe_2_SAL<- recipe_1_SAL %>% update_role(Date, new_role = 'ID')

recipe_1 %>% prep() %>% summary()
recipe_2 %>% prep() %>% summary()

recipe_1_SAL %>% prep() %>% summary()
recipe_2_SAL %>% prep() %>% summary()

# 6.0 Models ---- 

# Prophet w/ Regressor
wflw_fit_prophet<- workflow() %>% 
                   add_model(prophet_reg() %>% set_engine('prophet')) %>%
                   add_recipe(recipe_1) %>%
                   fit(training(splits))

wflw_fit_prophet_SAL<- workflow() %>% 
                       add_model(prophet_reg() %>% set_engine('prophet')) %>%
                       add_recipe(recipe_1_SAL) %>%
                       fit(training(splits_SAL))


# XGBoost
wflw_fit_xgboost<- workflow() %>% 
                   add_model(boost_tree() %>% set_engine('xgboost')) %>%
                   add_recipe(recipe_2) %>%
                   fit(training(splits))

wflw_fit_xgboost_SAL<- workflow() %>% 
                       add_model(boost_tree() %>% set_engine('xgboost')) %>%
                       add_recipe(recipe_2_SAL) %>%
                       fit(training(splits_SAL))

# Random Forest
wflw_fit_rf<- workflow() %>% 
              add_model(rand_forest() %>% set_engine('ranger')) %>%
              add_recipe(recipe_2) %>%
              fit(training(splits))


wflw_fit_rf_SAL<- workflow() %>% 
                  add_model(rand_forest() %>% set_engine('ranger')) %>%
                  add_recipe(recipe_2_SAL) %>%
                  fit(training(splits_SAL))

# SVM 
wflw_fit_svm<- workflow() %>% 
               add_model(svm_rbf() %>% set_engine('kernlab')) %>%
               add_recipe(recipe_2) %>%
               fit(training(splits))

wflw_fit_svm_SAL<- workflow() %>% 
                   add_model(svm_rbf() %>% set_engine('kernlab')) %>%
                   add_recipe(recipe_2_SAL) %>%
                   fit(training(splits_SAL))

# Prophet Boost
wflw_fit_prophet_boost<- workflow() %>% 
                         add_model(prophet_boost( seasonality_daily = FALSE, 
                                                  seasonality_weekly = FALSE, 
                                                  seasonality_yearly = FALSE) %>% set_engine('prophet_xgboost')) %>%
                         add_recipe(recipe_1) %>%
                         fit(training(splits))

wflw_fit_prophet_boost_SAL<- workflow() %>% 
                             add_model(prophet_boost( seasonality_daily = FALSE, 
                                                      seasonality_weekly = FALSE, 
                                                      seasonality_yearly = FALSE) %>% set_engine('prophet_xgboost')) %>%
                             add_recipe(recipe_1_SAL) %>%
                             fit(training(splits_SAL))



# 7.0 Modeltime workflow ----
submodels_tbl<- modeltime_table(wflw_fit_prophet,
                                wflw_fit_xgboost,
                                wflw_fit_rf,
                                wflw_fit_svm,
                                wflw_fit_prophet_boost)

submodels_tbl_SAL<- modeltime_table(wflw_fit_prophet_SAL,
                                    wflw_fit_xgboost_SAL,
                                    wflw_fit_rf_SAL,
                                    wflw_fit_svm_SAL,
                                    wflw_fit_prophet_boost_SAL)

# Calibrate testing data
submodel_calibrate_tbl<- submodels_tbl %>% modeltime_calibrate(testing(splits))

submodel_calibrate_tbl_SAL<- submodels_tbl_SAL %>% modeltime_calibrate(testing(splits_SAL))

# Measure Accuracy
perf_table_LT2<- submodel_calibrate_tbl %>% modeltime_accuracy()
perf_table_LT2

perf_table_SAL<- submodel_calibrate_tbl_SAL %>% modeltime_accuracy()
perf_table_SAL

#Visualize test forecast
submodel_calibrate_tbl %>% modeltime_forecast(new_data = testing(splits),
                                                  actual_data = train_LT2_tbl,
                                                  keep_data = TRUE) %>%
                           plot_modeltime_forecast()

submodel_calibrate_tbl_SAL %>% modeltime_forecast(new_data = testing(splits_SAL),
                                                  actual_data = train_SAL_tbl,
                                                  keep_data = TRUE) %>%
                               plot_modeltime_forecast()

#Refit
submodels_refit_tbl<- submodel_calibrate_tbl %>%
                      modeltime_refit(train_LT2_tbl)

submodels_refit_tbl_SAL<- submodel_calibrate_tbl_SAL %>%
                          modeltime_refit(train_SAL_tbl)

# Visualize refitted model
submodels_refit_tbl %>% modeltime_forecast(new_data = future_LT2_tbl,
                                           actual_data = train_LT2_tbl,
                                           keep_data = TRUE)%>%
                        plot_modeltime_forecast()

submodels_refit_tbl_SAL %>% modeltime_forecast(new_data = future_SAL_tbl,
                                               actual_data = train_SAL_tbl,
                                               keep_data = TRUE)%>%
                            plot_modeltime_forecast()


# 8.0 Ensemble models
top3_model_LT2<- perf_table_LT2 %>% top_n(-3, wt = rmse)
top3_model_SAL<- perf_table_SAL %>% top_n(-3, wt = rmse)


ensemble_fit_mean<- submodels_tbl %>%
                    filter(.model_id %in% top3_model_LT2$.model_id)  %>%
                    ensemble_average(type = 'mean')

loadings<- c(0.4,0.4,0.2)

ensemble_fit_weight<- submodels_tbl %>%
  filter(.model_id %in% top3_model_LT2$.model_id) 

ensemble_fit_weight<- ensemble_fit_weight[order(top3_model_LT2$rmse),]

ensemble_fit_weight<- ensemble_fit_weight %>% ensemble_weighted(loadings = c(0.4,0.4,0.2))

ensemble_tbl<- modeltime_table(ensemble_fit_mean)
ensemble_tbl_w<- modeltime_table(ensemble_fit_weight)


ensemble_tbl %>% combine_modeltime_tables(submodels_tbl) %>%
                 modeltime_accuracy(testing(splits))

ensemble_tbl_w %>% combine_modeltime_tables(submodels_tbl) %>% 
                   modeltime_accuracy(testing(splits))

ensemble_tbl %>% modeltime_forecast(new_data = testing(splits),
                                    actual_data = train_LT2_tbl,
                                    keep_data = TRUE) %>%
                 plot_modeltime_forecast()

ensemble_tbl_w %>% modeltime_forecast(new_data = testing(splits),
                                    actual_data = train_LT2_tbl,
                                    keep_data = TRUE) %>%
                   plot_modeltime_forecast()

ensemble_refit_tbl<- ensemble_tbl %>% modeltime_refit(train_LT2_tbl)

ensemble_refit_tbl %>% modeltime_forecast(new_data = future_LT2_tbl,
                                          actual_data = train_LT2_tbl,
                                          keep_data = TRUE) %>%
                      plot_modeltime_forecast()

#SAl ensemble
ensemble_fit_mean_SAL<- submodels_tbl_SAL %>%
                        filter(.model_id %in% top3_model_SAL$.model_id) %>%
                        ensemble_average(type = 'mean')
                        
                        
ensemble_fit_weight_SAL<- submodels_tbl_SAL %>%
                          filter(.model_id %in% top3_model_SAL$.model_id) 
                        
ensemble_fit_weight_SAL<- ensemble_fit_weight_SAL[order(top3_model_SAL$rmse),]
                        
ensemble_fit_weight_SAL<- ensemble_fit_weight_SAL %>% ensemble_weighted(loadings = c(0.4,0.4,0.2))


ensemble_tbl_SAL<- modeltime_table(ensemble_fit_mean_SAL)
ensemble_tbl_w_SAL<- modeltime_table(ensemble_fit_weight_SAL)


ensemble_tbl_SAL %>% combine_modeltime_tables(submodels_tbl_SAL) %>%
                     modeltime_accuracy(testing(splits_SAL))

ensemble_tbl_w_SAL %>% combine_modeltime_tables(submodels_tbl_SAL) %>%
                       modeltime_accuracy(testing(splits_SAL))

ensemble_tbl_SAL %>% modeltime_forecast(new_data = testing(splits_SAL),
                                        actual_data = train_SAL_tbl,
                                        keep_data = TRUE) %>%
                     plot_modeltime_forecast()


# 8.0 H2o modelling ----
library(h2o)
# full_data_LT2 %>% ggplot(aes(Date, Depth_to_Groundwater_LT2)) +
#                   #train region
#                   annotate('text', x = lubridate::ymd("2010-01-01"), y = -8.5,
#                   color = tidyquant::palette_light()[[1]], label = "Train Region") +
#                   #Validation region
#                   geom_rect(xmin = as.numeric(lubridate::ymd("2019-01-01")), 
#                   xmax = as.numeric(lubridate::ymd("2019-12-31")),
#                   ymin = 0, ymax = Inf, alpha = 0.02,
#                   fill = 'red') +
#                   annotate("text", x = lubridate::ymd("2019-02-01"), y = -8.5,
#                   color = tidyquant::palette_light()[[1]], label = "Validation\nRegion") +
#                   #test region
#                   geom_rect(xmin = as.numeric(lubridate::ymd("2020-01-01")), 
#                   xmax = as.numeric(lubridate::ymd("2020-06-30")),
#                   ymin = 0, ymax = Inf, alpha = 0.02,
#                   fill = tidyquant::palette_light()[[4]]) +
#                   annotate("text", x = lubridate::ymd("2020-06-01"), y = -8.5,
#                   color = tidyquant::palette_light()[[1]], label = "Test\nRegion") +
#                   #Data
#                   geom_line(col = tidyquant::palette_light()[1]) +
#                   # Aesthetics
#                   tidyquant::theme_tq() +
#                   scale_x_date(date_breaks = "1 year", date_labels = "%Y")
                  
  
                  


LT2_prep_h2o<- train_LT2_tbl %>% mutate(trend = 1:nrow(train_LT2_tbl), trend_sqr = trend^2)
LT2_prep_h2o<- LT2_prep_h2o %>% 
               tk_augment_timeseries_signature() %>% 
               select_if(~ !lubridate::is.Date(.)) %>%
               select_if(~ !any(is.na(.))) %>%
               mutate_if(is.ordered, ~ as.character(.) %>% as.factor)

LT2_train_h2o <- LT2_prep_h2o %>% filter(year < 2019)
LT2_valid_h2o <- LT2_prep_h2o %>% filter(year == 2019)
Lt2_test_h2o  <- LT2_prep_h2o %>% filter(year == 2020)

h2o.init()

train_h2o <- as.h2o(LT2_train_h2o)
valid_h2o <- as.h2o(LT2_valid_h2o)
test_h2o  <- as.h2o(Lt2_test_h2o)

y <- "Depth_to_Groundwater_LT2"
x <- setdiff(names(train_h2o), y)

automl_models_h2o <- h2o.automl(
  x = x, 
  y = y, 
  training_frame = train_h2o, 
  validation_frame = valid_h2o, 
  leaderboard_frame = test_h2o,
  stopping_tolerance = 0.005,
  max_runtime_secs = 600,
  stopping_metric = "RMSE")

automl_models_h2o@leaderboard

automl_leader <- automl_models_h2o@leader

automl_leader
library(vip)

vip_plot<- vip(automl_leader)

vip_plot

perf_automl<- h2o.performance(automl_leader, newdata = test_h2o)
perf_automl

path = "/models"

h2o.saveModel(automl_leader, path)

pred_h2o <- h2o.predict(automl_leader, newdata = test_h2o)


error_tbl <- train_LT2_tbl %>% 
  filter(lubridate::year(Date) == 2020) %>%
  add_column(pred = pred_h2o %>% as.tibble() %>% pull(predict)) %>%
  rename(actual = Depth_to_Groundwater_LT2)

error_tbl %>% ggplot(aes(Date)) +
              geom_line(aes(y = actual), col = 'blue', size = 1) +
              geom_line(aes(y = pred), col = 'red', size = 1) +
              ylab('')
