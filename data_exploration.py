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


def calculate_data_characteristics(data: pd.DataFrame) -> dict:
    """
    Calculates data characteristics.

    Args:
        data: The dataset in a Pandas DataFrame format.

    Returns:
        A dictionary containing the data characteristics.
    """

    # Define the metrics to be computed for each column
    metrics = ['min', '25_percentile', '50_percentile', '75_percentile', 'max', 'mean', 'std']

    characteristics = {}

    # Stats for the whole dataframe
    characteristics['dataframe_info'] = {
        "n_samples": data.shape[0],
        "n_regressors": data.shape[1] - 1  # Exclude the target variable
    }

    # Iterate over each column to compute statistics
    for column in data.columns:
        column_describe = data[column].describe(percentiles=[.25, .5, .75])

        # Convert Series to dictionary and replace '%' with 'percentile'
        column_describe_dict = column_describe.to_dict()
        column_describe_dict = {k.replace('%', 'percentile'): v for k, v in column_describe_dict.items()}

        column_stats = {}
        for metric in metrics:
            value = column_describe_dict.get(metric, None)
            if value is not None:
                value = round(value, 4)
            column_stats[metric] = value

        # Add column stats to the main dictionary
        characteristics[column] = column_stats

    return characteristics
