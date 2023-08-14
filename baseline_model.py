import itertools


import numpy as np
import pandas as pd
import CRPS.CRPS as pscore
from data_preparation import train_test_split
from mappings import map_date_to_month_id


def compute_baseline_tuning(data: pd.DataFrame, country_ids: list, forecast_month_set: list,
                            train_months_set: list, forecast_horizon_set: list,
                            quantiles_set: list, output_directory: str, year: int) -> tuple:
    """
    Compute the baseline tuning for a forecasting model using CRPS metric.
    To forecast a month t we compute the quantiles of w training months in a period [t-w-h, t-(h+1)] and compute the
    forecast at t = t-h. The forecast samples equal to the quantiles obtained previously and is evaluated using the CRPS
    metric.

    :param data: Dataset with conflict fatalities for all countries and all available data.
    :param country_ids: List of country IDs for which to compute the baseline tuning.
    :param forecast_month_set: List of months we want to forecast.
    :param train_months_set: List of training periods (in months) to consider.
    :param forecast_horizon_set: List of forecast horizons: indicates how far apart from forecast origin the forecast
    :param quantiles_set: List of lists of quantiles computed from training data as forecast sample
                                 is made (forecast month = forecast origin + forecast horizon)
    :param output_directory:
    :param year: identification of the year to safe the results under a corresponding name
    :return: A tuple containing two DataFrames:
             - First DataFrame: Average CRPS scores grouped by country, forecast_month, training months,
                                forecast horizon, quantiles, while averaging over the CRPS score.
             - Second DataFrame: Average CRPS scores grouped by forecast_month, training months, forecast horizon,
                               quantiles, while averaging over the CRPS score.
    """
    # Map datetime objects of forecast origins to month_ids
    forecast_month_set = [map_date_to_month_id(dt.year, dt.month) for dt in forecast_month_set]

    # Preallocate DataFrame memory
    combinations = list(itertools.product(country_ids, forecast_month_set, train_months_set, forecast_horizon_set,
                                          quantiles_set))
    crps_scores = pd.DataFrame(index=range(len(combinations)),
                               columns=['country_id', 'forecast_month', 'train_months', 'forecast_horizon',
                                        'quantiles', 'average_crps', 'average_fatalities'])

    # Iterate over each combination
    for i, (country, forecast_month, train_months, forecast_horizon, quantiles) in enumerate(combinations):
        data_country = data[data['country_id'] == country]

        # Split data set into training and evaluation data
        train_set, eval_set = train_test_split(data=data, country_id=country, forecast_month=forecast_month,
                                               train_months=train_months, forecast_horizon=forecast_horizon)

        # Compute average conflict fatalities in evaluation_set
        average_conflict_fatalities = eval_set['ged_sb'].mean()

        # Call the compute_baseline_crps function
        crps = compute_baseline_crps(train_set=train_set, eval_set=eval_set, quantiles=quantiles)

        crps_scores.loc[i] = {
            'country_id': country,
            'forecast_month': forecast_month,
            'train_months': train_months,
            'forecast_horizon': forecast_horizon,
            'quantiles': quantiles,
            'average_crps': crps,
            'average_fatalities': average_conflict_fatalities
        }
        # Store the results in the DataFrame

    # Convert the list "quantiles" to a tuple to have a hashable type for the groupby command
    crps_scores['quantiles'] = crps_scores['quantiles'].apply(tuple)

    # year_average_crps by ['country_id', 'train_months', 'forecast_horizon', 'quantiles']
    crps_scores_all_year_country_specific = crps_scores.groupby([
        'country_id', 'train_months', 'forecast_horizon', 'quantiles']).mean()[
        ['average_crps', 'average_fatalities']].reset_index()

    # year_average_crps by ['train_months', 'forecast_horizon', 'quantiles']
    crps_scores_all_year_global = crps_scores.groupby(['train_months', 'forecast_horizon', 'quantiles']).mean()[
        ['average_crps', 'average_fatalities']].reset_index()

    # Convert quantiles as tuple type back to list such that it can be sorted and filtered by specific lists
    crps_scores_all_year_country_specific['quantiles'] = crps_scores_all_year_country_specific['quantiles'].apply(list)
    crps_scores_all_year_global['quantiles'] = crps_scores_all_year_global['quantiles'].apply(list)
    # Write crps_scores_country_specific as a parquet file
    crps_scores_all_year_country_specific.to_parquet(output_directory + f"/crps_scores_all_year_{year}_country_specific.parquet")

    # Write crps_scores_global as a parquet file
    crps_scores_all_year_global.to_parquet(output_directory + f"/crps_scores_all_year_{year}_global.parquet")

    # Write crps_scores in a parquet file
    crps_scores.to_parquet(output_directory + f"/crps_scores_combinations_all_year_{year}.parquet")

    return crps_scores_all_year_country_specific, crps_scores_all_year_global


def compute_baseline_crps(train_set: pd.DataFrame, eval_set: pd.DataFrame, quantiles: list) -> float:
    """
    Performs forecast for the specific model settings and returns the CRPS value of the prediction for forecast_month

    :param eval_set: evaluation set respective to evaluation months
    :param train_set: training set respective to forecast horizon and training months
    :param quantiles: quantiles we want to compute as forecast sample
    :return: CRPS value of the prediction for forecasting month
    """

    # Compute quantiles of training data for the specified quantiles
    quantiles = compute_quantiles(train_set, quantiles)

    # Compute forecast samples based on quantiles computed from training data
    crps = compute_crps_baseline(eval_data=eval_set, quantiles=quantiles)

    return crps


def compute_quantiles(train_data: pd.DataFrame, quantiles: list) -> pd.DataFrame:
    """
    Function to compute quantiles of training data for specified quantiles
    :param train_data: conflict fatalities of specified training period
    :param quantiles: quantiles we want to compute
    :return: data frame with quantiles and the corresponding values
    """
    # Setting up data frame
    quantiles_df = pd.DataFrame({'quantile': quantiles, 'quantile_values': np.nan})

    # Determining quantile values for the requested quantiles
    quantile_values = np.quantile(train_data['ged_sb'], quantiles)
    quantiles_df['quantile_values'] = quantile_values

    return quantiles_df


def compute_crps_baseline(eval_data: pd.DataFrame, quantiles: pd.DataFrame) -> float:
    """
    Compute CRPS score based on quantiles as forecasting sample for forecasting month
    :param eval_data: evaluation data corresponding forecast_month
    :param quantiles: Quantiles with corresponding quantile values
    :return: CRPS score forecasting month
    """
    # Compute CRPS score based on quantile sample for forecast_month
    observation = eval_data['ged_sb'].iloc[0]
    quantile_values = np.array(quantiles['quantile_values'])
    crps, _, _ = pscore(quantile_values, observation).compute()

    return crps

