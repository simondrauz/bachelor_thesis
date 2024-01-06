import itertools
from typing import Tuple

import CRPS.CRPS as pscore
import pandas as pd

from data_preparation import train_test_split
from mappings import map_date_to_month_id


# Note: This function has to be updated to the bayesian model
def compute_bayesian_model(data: pd.DataFrame, target_variable: str, predictors: list, sample_iterations_set: list,
                           country_ids: list, forecast_month_set: list, train_months_set: list,
                           forecast_horizon_set: list, output_directory: str, year: int) \
        -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute the bayesian model tuning for a forecasting model using CRPS metric.

    :param data: Dataset with conflict fatalities for all countries and all available data.
    :param target_variable: variable we want to forecast
    :param predictors: variables we want to use as predictors
    :param sample_iterations_set: List of iterations to sample from the posterior distribution
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
    combinations = list(itertools.product(country_ids, forecast_month_set, train_months_set, forecast_horizon_set,
                                          sample_iterations_set))
    crps_scores = pd.DataFrame(index=range(len(combinations)),
                               columns=['country_id', 'forecast_month', 'train_months', 'forecast_horizon',
                                        'sample_iterations', 'average_crps', 'average_fatalities'])

    # Iterate over each combination
    for i, (country, forecast_month, train_months, forecast_horizon, sample_iterations) in enumerate(combinations):
        data_country = data[data['country_id'] == country]

        # Split data set into training and evaluation data
        train_set, eval_set = train_test_split(data=data, country_id=country, forecast_month=forecast_month,
                                               train_months=train_months, forecast_horizon=forecast_horizon)

        # Compute average conflict fatalities in evaluation_set
        average_conflict_fatalities = eval_set['ged_sb'].mean()

        # Build the Bayesian model
        model, trace = build_bayesian_model(df=train_set, target_variable=target_variable, predictors=predictors,
                                            sample_iterations=sample_iterations)
        # Call the compute_baseline_crps function
        crps = compute_crps_bayesian(eval_data=eval_set, trace=trace)

        crps_scores.loc[i] = {
            'country_id': country,
            'forecast_month': forecast_month,
            'train_months': train_months,
            'forecast_horizon': forecast_horizon,
            'sample_iterations': sample_iterations,
            'average_crps': crps,
            'average_fatalities': average_conflict_fatalities
        }
        # Store the results in the DataFrame

    # Note: for train_months = 'all', the actual no. of training months will be displayed
    # year_average_crps by ['country_id', 'train_months', 'forecast_horizon', 'quantiles']
    crps_scores_all_year_country_specific = crps_scores.groupby([
        'country_id', 'train_months', 'forecast_horizon', 'sample_iterations']).mean()[
        ['average_crps', 'average_fatalities']].reset_index()

    # year_average_crps by ['train_months', 'forecast_horizon', 'quantiles']
    crps_scores_all_year_global = crps_scores.groupby(['train_months', 'forecast_horizon', 'sample_iterations']).mean()[
        ['average_crps', 'average_fatalities']].reset_index()

    # Write crps_scores_country_specific as a parquet file
    crps_scores_all_year_country_specific.to_parquet(output_directory + f"/crps_scores_all_year_{year}_country_specific.parquet")

    # Write crps_scores_global as a parquet file
    crps_scores_all_year_global.to_parquet(output_directory + f"/crps_scores_all_year_{year}_global.parquet")

    # Write crps_scores in a parquet file
    crps_scores.to_parquet(output_directory + f"/crps_scores_combinations_all_year_{year}.parquet")

    return crps_scores_all_year_country_specific, crps_scores_all_year_global


def build_bayesian_model(df: pd.DataFrame, target_variable: str, predictors: list, sample_iterations: int) \
        -> Tuple[pm.Model, pm.backends.base.MultiTrace]:
    """
    Builds a Bayesian linear regression model for a country and forecasting month using data of a specific country and
    specified time as training period.

    Parameters:
        df (pandas.DataFrame): The DataFrame containing the data.
        target_variable (str): The name of the target variable.
        predictors (list[str]): A list of input variable names.
        sample_iterations (int): The number of samples drawn from the posterior distribution.

    Returns:
        tuple[pymc3.Model, pymc3.backends.base.InferenceData]: The built Bayesian model and the inference data.

    """
    # Select the target variable and input variables from the DataFrame
    y = df[target_variable]
    X = df[predictors]
    # Define dictionary for variable names
    COORDS = {"predictors": X.columns.values}

    # Build the model
    with pm.Model(coords=COORDS) as model:
        # Define the priors
        alpha = pm.Normal('alpha', mu=0, sigma=10)
        beta = pm.Normal('beta', mu=0, sigma=10, dims='predictors')

        # Prior distribution for the negative binomial overdispersion parameter
        alpha_nb = pm.Exponential('alpha_nb', 0.5)

        # sigma = pm.HalfNormal('sigma', sigma=1)
        # n = (mu*mu/(sigma*sigma-mu))
        # p = mu-(sigma*sigma)

        # Define the likelihood
        mu = pm.math.exp(alpha + pm.math.dot(beta, X.values.T))

        likelihood = pm.NegativeBinomial('likelihood', mu=mu, alpha=alpha_nb, observed=y.values)

        # Note: Optionally define cores and chains
        # Run the sampling using the No-U-Turn Sampler (NUTS) for the specified number of samples
        trace = pm.sample(sample_iterations, return_inferencedata=True)

    return model, trace


def compute_crps_bayesian(eval_data: pd.DataFrame, trace: pm.backends.base.MultiTrace) -> float:
    """
    Compute CRPS score based on samples as forecasting sample for forecasting month
    :param eval_data: evaluation data corresponding forecast_month
    :param trace: trace object of bayesian model
    :return: CRPS score forecasting month
    """
    # Extract samples obtained from the posterior distribution of the likelihood from the trace
    samples = trace.posterior['likelihood'].values

    # Compute CRPS score based on samples for forecast_month
    observation = eval_data['ged_sb'].iloc[0]
    crps, _, _ = pscore(samples, observation).compute()

    return crps
