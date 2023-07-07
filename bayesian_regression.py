import pymc3 as pm
import numpy as np
import pandas as pd
import itertools

import int
import numpy as np
import pandas as pd
import CRPS.CRPS as pscore

from mappings import map_date_to_month_id
from data_preparation import train_test_split

# Note: This function has to be updated to the bayesian model
def compute_bayesian_model(data: pd.DataFrame, country_ids: list[int], forecast_month_set: list,
                           train_months_set: list[int], forecast_horizon_set: list, output_directory: str, year: int) \
        -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute the bayesian model tuning for a forecasting model using CRPS metric.

    :param data: Dataset with conflict fatalities for all countries and all available data.
    :param country_ids: List of country IDs for which to compute the baseline tuning.
    :param forecast_month_set: List of months we want to forecast.
    :param train_months_set: List of training periods (in months) to consider.
    :param forecast_horizon_set: List of forecast horizons: indicates how far apart from forecast origin the forecast
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
    combinations = list(itertools.product(country_ids, forecast_month_set, train_months_set, forecast_horizon_set))
    crps_scores = pd.DataFrame(index=range(len(combinations)),
                               columns=['country_id', 'forecast_month', 'train_months', 'forecast_horizon',
                                        'average_crps', 'average_fatalities'])

    # Iterate over each combination
    for i, (country, forecast_month, train_months, forecast_horizon) in enumerate(combinations):
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


def build_bayesian_model(df: pd.DataFrame, target_variable: str, input_variables: list[str], sample_iterations: int) \
        -> tuple[pm.Model, pm.backends.base.InferenceData]:
    """
    Builds a Bayesian linear regression model for a country and forecasting month using data of a specific country and
    specified time as training period.

    Parameters:
        df (pandas.DataFrame): The DataFrame containing the data.
        target_variable (str): The name of the target variable.
        input_variables (list): A list of input variable names.
        :param sample_iterations: number of samples drawn from the posterior distribution

    Returns:
        pymc3.Model: The built Bayesian model.
    """
    # Select the target variable and input variables from the DataFrame
    variables = [target_variable] + input_variables
    df = df[variables]

    # Build the model
    with pm.Model() as model:
        # Define the priors
        alpha = pm.Normal('alpha', mu=0, sigma=10)
        beta = pm.Normal('beta', mu=0, sigma=10, shape=len(input_variables))
        sigma = pm.HalfNormal('sigma', sigma=1)

        # Define the likelihood
        mu = alpha + pm.math.dot(beta, df[input_variables].values.T)
        likelihood = pm.Normal('likelihood', mu=mu, sigma=sigma, observed=df[target_variable].values)

        # Run the sampling using the No-U-Turn Sampler (NUTS) for n samples
        trace = pm.sample(sample_iterations, chains=4, cores=4, return_inferencedata=True)
    return model, trace
