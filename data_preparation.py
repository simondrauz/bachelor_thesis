from typing import Tuple, List

import datetime
import patsy as ps
import numpy as np
import pandas as pd
from pandas import DataFrame

from mappings import import_country_mapping


def select_data(df: pd.DataFrame, country_id: np.integer, month_cut: int) -> pd.DataFrame:
    """
    Selects data for requested country and time period
    :param df: data of all countries and whole time period
    :param country_id: specifies country the data should be filtered on
    :param month_cut: the data from this month on and more recent will be included
    :return: data for requested country and time period
    """
    # Filter data by country
    df = df[(df["country_id"] == country_id) & (df['month_id'] >= month_cut)]

    # Remove duplicate 'month_id' values by taking the smaller value of the corresponding target variable
    df = df.sort_values('month_id')  # Sort the DataFrame by 'month_id'

    return df


def prepare_data(df: pd.DataFrame, variables_to_check: list, duplicate_check=True) -> pd.DataFrame:
    """
    Prepares the data such that duplicate values are found and aggregated by keeping the smaller value
    and additionally filling the NaN values with 0s.
    Optionally there's the possibility to examine the duplicate values. Check the 'check_duplicates' doc for further
    information.
    :param df: data frame of data set with or without features
    :param variables_to_check: determines which for which variables of the data frame duplicates should be examined
    :param duplicate_check: determines if the nature of the duplicates should be analysed
    :return: data frame of data set with duplicates handled
    """
    # Find duplicate rows regarding 'month_id' and 'country_id'
    duplicates = find_duplicates(df)

    if duplicate_check is True:
        # Check the nature of the duplicates
        duplicate_check_result = check_duplicates(duplicates, variables_to_check)
        print(duplicate_check_result)

    # Group duplicate rows
    df = replace_duplicates(df, duplicates)

    # Replace NaN values in the target variables with 0
    df = fill_na_values(df)

    return df


def find_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Scans the given dataframe for duplicates in the columns ['month_id', 'country_id'] and returns a data frame
    with the corresponding rows
    :param df: data frame with conflict fatalities data of the defined country and time period
    :return: data frame with the corresponding duplicate rows
    """
    duplicates_ind = df.duplicated(subset=['country_id', 'month_id'], keep=False)
    duplicates = df[duplicates_ind]

    return duplicates


def list_contains_nan(lst: np.ndarray) -> bool:
    """
    Checks if any value of the list is a nan value
    :param lst: List with floats or nan values
    :return: boolean indicating if the list contains a nan value or not
    """
    return np.isnan(np.sum(lst))


# ToDo: Look into duplicates (e.g. exception if different values at duplicate)
# ToDO: Might has to be revised for duplicate checks on whole dataframe ->there seem to be no duplicates now??
def check_duplicates(df, variables_to_check: list) -> pd.DataFrame:
    """
    Checks the nature of the found duplicates for each of the conflict fatalitiy counts. We differenciate:
    Only NaN: The found duplicate has a nan value for both entries.
    Value and NaN: One of the entries is a value, the other one is a NaN
    Same value: Both entries of the duplicate are a value and identical
    Different values: The entries of the duplicate differ from each other
    Different Values and NaN: In the "duplicate" are different values as well as NaN values

    :param variables_to_check: variables that should be checked for duplicates
    :param df: data frame with duplicate rows only
    :return: data frame that defines the nature of the found duplicates
    """
    # Group the DataFrame by 'country_id' and 'month_id'
    grouped = df.groupby(['country_id', 'month_id'])

    # Initialize an empty DataFrame to store the results
    results = pd.DataFrame(columns=['country_id', 'month_id'])
    for column in variables_to_check:
        results[f'check_{column}'] = []

    # Iterate over each group
    for (country_id, month_id), group in grouped:
        result_row = pd.Series({'country_id': country_id, 'month_id': month_id})

        # Iterate over each column dynamically
        for column in variables_to_check:
            unique_values = group[column].unique()
            unique_values = np.unique(np.round(np.array(unique_values).astype(float), decimals=0))

            if len(unique_values) == 1:
                if list_contains_nan(unique_values):
                    result_row[f'check_{column}'] = 'Only NaN'
                else:
                    result_row[f'check_{column}'] = 'Same value'
            elif len(unique_values) == 2:
                if list_contains_nan(unique_values):
                    result_row[f'check_{column}'] = 'Value and NaN'
                else:
                    result_row[f'check_{column}'] = 'Different values'
            else:
                if list_contains_nan(unique_values):
                    result_row[f'check_{column}'] = 'Different Values and NaN'
                else:
                    result_row[f'check_{column}'] = 'Different values'

        # Append the result to the 'results' DataFrame
        results = pd.concat([results, result_row], ignore_index=True)

    return results
    # Note: Previous old version for only ['ged_sb', 'ged_ns', 'ged_os']
    # # Iterate over each group
    # for (country_id, month_id), group in grouped:
    #     # Check 'ged_sb' column
    #     unique_values_sb = group['ged_sb'].unique()
    #     unique_values_sb = np.unique(np.round(np.array(unique_values_sb).astype(float), decimals=0))
    #     if len(unique_values_sb) == 1:
    #         if list_contains_nan(unique_values_sb):
    #             check_ged_sb = 'Only NaN'
    #         else:
    #             check_ged_sb = 'Same value'
    #     elif len(unique_values_sb) == 2:
    #         if list_contains_nan(unique_values_sb):
    #             check_ged_sb = 'Value and NaN'
    #         else:
    #             check_ged_sb = 'Different values'
    #     else:
    #         if list_contains_nan(unique_values_sb):
    #             check_ged_sb = 'Different Values and NaN'
    #         else:
    #             check_ged_sb = 'Different values'
    #
    #     # Check 'ged_ns' column
    #     unique_values_ns = group['ged_ns'].unique()
    #     unique_values_ns = np.unique(np.round(np.array(unique_values_ns).astype(float), decimals=0))
    #     if len(unique_values_ns) == 1:
    #         if list_contains_nan(unique_values_ns):
    #             check_ged_ns = 'Only NaN'
    #         else:
    #             check_ged_ns = 'Same value'
    #     elif len(unique_values_ns) == 2:
    #         if list_contains_nan(unique_values_ns):
    #             check_ged_ns = 'Value and NaN'
    #         else:
    #             check_ged_ns = 'Different values'
    #     else:
    #         if list_contains_nan(unique_values_ns):
    #             check_ged_ns = 'Diffrent Values and NaN'
    #         else:
    #             check_ged_ns = 'Different values'
    #
    #     # Check 'ged_os' column
    #     unique_values_os = group['ged_os'].unique()
    #     unique_values_os = np.unique(np.round(np.array(unique_values_os).astype(float), decimals=0))
    #     if len(unique_values_os) == 1:
    #         if list_contains_nan(unique_values_os):
    #             check_ged_os = 'Only NaN'
    #         else:
    #             check_ged_os = 'Same value'
    #     elif len(unique_values_os) == 2:
    #         if list_contains_nan(unique_values_os):
    #             check_ged_os = 'Value and NaN'
    #         else:
    #             check_ged_os = 'Different values'
    #     else:
    #         if list_contains_nan(unique_values_os):
    #             check_ged_os = 'Diffrent Values and NaN'
    #         else:
    #             check_ged_os = 'Different values'
    #
    #     # Append the result to the 'results' DataFrame
    #     results = results.append({
    #         'country_id': country_id,
    #         'month_id': month_id,
    #         'check_ged_sb': check_ged_sb,
    #         'check_ged_ns': check_ged_ns,
    #         'check_ged_os': check_ged_os
    #     }, ignore_index=True)
    #
    # return results


def replace_duplicates(df: pd.DataFrame, df_duplicates: pd.DataFrame) -> pd.DataFrame:
    """
    Groups duplicate values in the column 'month_id' by taking the minimum value of the other columns
    :param df: data frame with data of the defined country and time period
    :param df_duplicates: data frame with duplicates corrected
    :return:
    """
    duplicated_values = df_duplicates.drop_duplicates(subset=['country_id', 'month_id'])[
        ['country_id', 'month_id']].values.tolist()
    print(
        f'Replacing the value of {len(duplicated_values)} duplicates at {duplicated_values} with the smaller value of conflict fatalities respective the type of conflict')

    # Take the minimum value for each 'month_id'
    df = df.groupby(['country_id', 'month_id']).min().reset_index()
    return df


def fill_na_values(df: pd.DataFrame):
    """
    Tracks the number of NaN values in the conflict fatality counts and replaces them with 0
    :param df: data frame with data of the defined country and time period
    :return: data frame with NaN values filled
    """
    # Get column names of dataframe
    columns = df.columns.tolist()
    columns.remove('country_id')
    columns.remove('month_id')

    # Fill NaN values of each column with 0s
    for column in columns:
        na_values_column = df[column].isna().sum()
        print(f'Replacing {na_values_column} NaN values in {column} with 0')
        df[column] = df[column].fillna(0)

    return df


# ToDo: Check if needed, of needed revise
def train_test_split(data: pd.DataFrame, country_id: int, forecast_month: int, train_months: int, forecast_horizon: int
                     ) -> (pd.DataFrame, pd.DataFrame):
    """
    Splits data set into training data and evaluation data.
    :param data: data set with conflict fatalities of all countries and all available data, respectively
    :param country_id: country_id of the country of interest
    :param forecast_month: month we want to compute forecast for
    :param train_months: number of months used in training period
    :param forecast_horizon: number of months the forecast is made into the futute from the last training month
    :return: training set and evaluation set
    """

    # Handle special case of all available data used for training
    if train_months == 'all':
        all_months = data['month_id'].unique()
        all_months_previous = all_months[all_months < forecast_month]
        train_months = len(all_months_previous)

    # Calculate start month of training period
    start_month_train = forecast_month - (train_months + forecast_horizon - 1)

    # Filter data such that necessary months for training and evaluation are included for a specific country
    data = data[(data['country_id'] == country_id) &
                (data['month_id'] >= start_month_train) &
                (data['month_id'] <= forecast_month)]

    # Specify training and evaluation data (evaluation data is basically one month, historically grown)
    train_set = data[data['month_id'] <= (forecast_month - forecast_horizon)]
    eval_set = data[data['month_id'] == forecast_month]

    return train_set, eval_set


def preprocess_data(df: pd.DataFrame, predictors: List[str], target_variable: str, country_name: str
                    ) -> Tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
    """
    Takes a dataframe, predictors, target_variable and country_name.
    Returns four modified dataframes:
    - df with non-standardized target and standardized predictors
    - df with standardized target and standardized predictors
    - df with non-standardized target and logarithm of standardized predictors
    - df with standardized target and logarithm of standardized predictors
    """

    # Extract country mapping and get country ID
    country_mapping = import_country_mapping()
    country_id = country_mapping.loc[country_mapping['name'] == country_name, 'country_id'].iloc[0]

    # Extract target and predictor data for the given country
    X = df.loc[df['country_id'] == country_id, predictors].reset_index(drop=True, inplace=False)
    Y = df.loc[df['country_id'] == country_id, target_variable].reset_index(drop=True, inplace=False)

    # Standardize
    Y_std = standardize(Y)
    X_std = standardize(X)

    # Log-transform standardized predictors
    X_std_log = take_logarithm(X_std)

    # Create modified dataframes
    # Create data frame with target variable and standardized input data
    df_Y_non_std_X_std = pd.concat([Y, X_std], axis=1)
    # Create data frame with standardized target variable and standardized input data
    df_Y_std_X_std = pd.concat([Y_std, X_std], axis=1)
    # Create data frame with target variable and standardized logarithm of input data
    df_Y_non_std_X_std_log = pd.concat([Y, X_std_log], axis=1)
    # Create data frame with standardized target variable and standardized logarithm of input data
    df_Y_std_X_std_log = pd.concat([Y_std, X_std_log], axis=1)

    return df_Y_non_std_X_std, df_Y_std_X_std, df_Y_non_std_X_std_log, df_Y_std_X_std_log


def standardize(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize a pandas DataFrame"""
    standardized_df = pd.DataFrame()
    for column in df.columns:
        standardized_column = (df[column] - df[column].mean()) / df[column].std()
        standardized_df[column] = standardized_column
    return standardized_df


def take_logarithm(df: pd.DataFrame) -> pd.DataFrame:
    """take logarithm a pandas DataFrame"""
    logarithmized_df = pd.DataFrame()
    for column in df.columns:
        logarithmized_column = np.log(df[column])
        logarithmized_df[column] = logarithmized_column
    return logarithmized_df


# Note: So far this function is not used
def add_year_column(df: pd.DataFrame) -> pd.DataFrame:
    # Extracting year from month_id by dividing by 12 and adding the base year 1990
    df['year'] = ((df['month_id'] - 1) // 12) + 1990
    return df


# ToDo: Check functionality of quantile-based knot generation
def generate_knots(data, num_knots, equal_spacing=True, quantile_spacing=False):
    """
    Generate knot locations for spline basis functions.

    Args:
    - data (array-like): Data on which to base the knot locations.
    - num_knots (int): Number of knots to generate.
    - method (str): Method to use for generating knots. Options are "equal_spacing" and "quantiles".

    Returns:
    - knots (np.array): Array of knot locations.
    """

    if equal_spacing:
        # Equal spacing method
        knots = np.linspace(data.min(), data.max(), num_knots + 2)[1:-1]
    elif quantile_spacing:
        # Quantiles method
        quantiles = np.linspace(0, 1, num_knots + 2)[1:-1]
        knots = np.quantile(data, quantiles)

    return knots


def generate_spline_design_matrix(covariate_data: np.ndarray, knots: np.ndarray, spline_degree: int) -> ps.DesignMatrix:
    """
    Generate a spline design matrix using the specified knot locations and spline degree.

    Args:
    - covariate_data (array-like): Data on which to base the design matrix.
    - knots (array-like): Locations of the interior knots.
    - spline_degree (int): Degree of the spline.

    Returns:
    - design_matrix (patsy DesignMatrix): Spline-based design matrix.

    """

    design_matrix = ps.bs(covariate_data, knots=knots, degree=spline_degree, include_intercept=False)

    return design_matrix


def get_training_data(df: pd.DataFrame, evaluation_year: int, forecast_horizon=3) -> pd.DataFrame:
    """
    Returns the training data for a specific evaluation year based on a forecast horizon of three months
    Args:
        df: preprocessed data frame
        evaluation_year: year we want to obtain forecasts eventually
        forecast_horizon: value of three months taken in accordance with competition setting

    Returns: truncated data frame
    """
    # Calculate the last month of training based on evaluation year and forecast horizon
    # Set the month to October of the previous year (given the horizon of 3 months)
    last_training_month = datetime(evaluation_year - 1, 13 - forecast_horizon, 1)

    # Filter dataframe using datetime comparison
    training_data = df[df['month_id'] <= last_training_month]

    return training_data


def generate_data_dict_pre_modelling(data_transformation: str,
                                     predictors: list,
                                     target_variable: str
                                     ) -> dict:
    """
    Generates a dictionary summarizing the key properties of the model input
    Args:
        data_transformation:
        predictors:
        target_variable:

    Returns:

    """
    return {
        'data_transformation': data_transformation,
        'predictors': predictors,
        'target_variable': target_variable
    }
