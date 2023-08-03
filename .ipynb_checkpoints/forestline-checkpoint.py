import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor

def get_forest(df, predict_var, response_var, trees=5000,
               grid_min=0, grid_max=3, grid=0.001, sample_prop=0.5, depth=1,
               replace=False):
    """
    Performs random forest regression and returns predictions.

    Args:
    - df (DataFrame): Input dataframe.
    - predict_var (str): Name of the predictor variable.
    - response_var (str): Name of the response variable.
    - trees (int, optional): Number of trees in the random forest. Defaults to 5000.
    - grid_min (float, optional): Minimum value for the predictor grid. Defaults to 0.
    - grid_max (float, optional): Maximum value for the predictor grid. Defaults to 3.
    - grid (float, optional): Step size for the predictor grid. Defaults to 0.001.
    - sample_prop (float, optional): Proportion of samples to use for each tree. Defaults to 0.5.
    - depth (int, optional): Maximum depth of each tree. Defaults to 1.
    - replace (bool, optional): Whether to use sampling with replacement. Defaults to False.

    Returns:
    - DataFrame: Predictions for the predictor grid.
    """
    # Select the predictor and response variables from the dataframe
    df = df[[predict_var, response_var]]

    # Capture the response variable name
    resp = response_var

    # Create a predictor grid
    pred_grid = pd.DataFrame({predict_var: np.arange(grid_min, grid_max, grid)})

    # Build the random forest regression model
    model = RandomForestRegressor(n_estimators=trees, max_depth=depth)
    model.fit(df[[predict_var]], df[response_var])

    # Predict the mean response using the model and predictor grid
    pred_mean = pd.DataFrame({'mean': model.predict(pred_grid)})

    # Concatenate the predictor grid, mean predictions, and quantile predictions
    df_result = pd.concat([pred_grid, pred_mean], axis=1)

    # Rename the columns to match the variable names
    df_result.rename(columns={predict_var: predict_var}, inplace=True)
    df_result.rename(columns={col: response_var + '_' + col for col in df_result.columns
                              if col == 'mean'}, inplace=True)

    return df_result
