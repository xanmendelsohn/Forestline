#################################################################################################################
## This function takes a dataframe (df) along with the names of the predictor variable (predict_var) and response 
## variable (response_var). It builds a random forest model using the ranger package.
## There are various customizable parameters such as the number of trees (trees), quantiles to predict (quantiles), 
## and grid specifications for the predictor variable.
## The code selects the predictor and response variables from the dataframe, creates a grid of predictor values, and 
## fits the random forest model. It then uses the model to predict the mean response and specified quantiles for the 
## predictor grid.
## Finally, the code combines the predictor grid, mean predictions, and quantile predictions into a single dataframe, 
## with appropriate renaming of columns. The resulting dataframe is returned as the output of the function.
## Please note that this code assumes the required packages (dplyr, stringr, and ranger) are already imported or 
## available in the environment.
#################################################################################################################

get_forest <- function(df, predict_var, response_var, trees = 5000, quantiles = c(0.1, 0.5, 0.9),
                       grid_min = 0, grid_max = 3, grid = 0.001, sample_prop = 0.5, depth = 1,
                       replace = FALSE) {
  # Select the predictor and response variables from the dataframe
  df <- df %>% select(predictor = {{predict_var}}, response = {{response_var}})
  
  # Capture the response variable name
  resp <- rlang::enquo(response_var) %>% rlang::as_label()
  
  # Create a grid of predictor values
  pred_grid <- tibble(predictor = seq(grid_min, grid_max, by = grid))
  
  # Build the random forest model using ranger
  model <- ranger::ranger(response ~ ., data = df, replace = replace, num.trees = trees,
                          quantreg = TRUE, sample.fraction = sample_prop, max.depth = depth)
  
  # Predict the mean response using the model and predictor grid
  pred_mean <- predict(model, data = pred_grid, type = 'response')$predictions %>% 
    tibble(mean = .)
  
  # Predict the specified quantiles using the model and predictor grid
  pred_quantile <- predict(model, data = pred_grid, predict.all = TRUE, type = 'quantiles',
                           quantiles = quantiles)$predictions %>% as_tibble() %>%
    dplyr::rename_with(., ~ stringr::str_replace(., 'quantile= ', 'q_'))
  
  # Combine the predictor grid, mean predictions, and quantile predictions into a single dataframe
  df_result <- dplyr::bind_cols(pred_grid, pred_mean, pred_quantile) %>%
    rename("{{predict_var}}" := predictor) %>% 
    dplyr::rename_with(.fn = ~ stringr::str_c(resp, '_', .), .cols = matches('^(q_|mean)'))
  
  # Return the resulting dataframe
  return(df_result)
}





