import pandas as pd


def calculate_correlation(data: pd.DataFrame, target_variables: list, input_variables: list):
    """
    Calculate the correlation between a list of target variables and a list of input variables in a pandas DataFrame.

    Parameters:
        data (pandas.DataFrame): The DataFrame containing the variables.
        target_variables (list): A list of target variable names.
        input_variables (list): A list of input variable names.

    Returns:
        pandas.DataFrame: A DataFrame containing the correlation values.
    """
    # Select the target variables and input variables from the DataFrame
    variables = target_variables + input_variables
    df = data[variables]

    # Calculate the correlation matrix
    correlation_matrix = df.corr()

    # Extract the correlation values of the target variables
    correlation_values = correlation_matrix.loc[target_variables]

    # Remove the target variables from the correlation values
    correlation_values = correlation_values.drop(columns=target_variables)

    return correlation_values

