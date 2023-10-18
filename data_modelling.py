import json
import time
from typing import Tuple, Optional, Union, Dict, Any

import CRPS.CRPS as pscore
import arviz as az
import numpy as np
import optuna
import pandas as pd
import pystan
from numpy import ndarray
from optuna.samplers import GridSampler
from pandas import DataFrame

from data_logging import (mlflow_logging_optuna_bayesian,
                          mlflow_logging_optuna_baseline, ml_flow_logging_actual_baseline)
from help_functions import seconds_to_format


class HyperparameterTuning:
    def __init__(self,
                 actuals: pd.DataFrame,
                 training_data: pd.DataFrame,
                 k_fold_iterations: int,
                 forecast_horizon: int,
                 target_variable: str,
                 covariates_dict: Optional[Dict] = None,
                 rolling_window_length: Optional[int] = None,
                 stan_model: Optional[pystan.StanModel] = None,
                 data_generation_specification: Optional[Dict] = None,
                 model_hyperparameter_settings: Optional[Dict] = None,
                 cross_validation_settings: Optional[Dict] = None,
                 sampling_parameters: Optional[Dict] = None,
                 validate_on_horizon: Optional[bool] = None,
                 validate_on_horizon_and_month: Optional[bool] = None,
                 is_bayesian_model: Optional[bool] = None,
                 is_baseline_model: Optional[bool] = None) -> None:
        self.actuals = actuals
        self.training_data = training_data
        self.k_fold_iterations = k_fold_iterations
        self.forecast_horizon = forecast_horizon
        self.target_variable = target_variable
        self.covariates_dict = covariates_dict
        self.rolling_window_length = rolling_window_length
        self.stan_model = stan_model
        self.data_generation_specification = data_generation_specification
        self.model_hyperparameter_settings = model_hyperparameter_settings
        self.cross_validation_settings = cross_validation_settings
        self.sampling_parameters = sampling_parameters
        self.validate_on_horizon = validate_on_horizon
        self.validate_on_horizon_and_month = validate_on_horizon_and_month
        self.is_bayesian_model = is_bayesian_model
        self.is_baseline_model = is_baseline_model

    def objective(self, trial: optuna.Trial) -> float:
        crps_score = run_hyperparameter_tuning(
            trial,
            actuals=self.actuals,
            training_data=self.training_data,
            k_fold_iterations=self.k_fold_iterations,
            forecast_horizon=self.forecast_horizon,
            target_variable=self.target_variable,
            covariates_dict=self.covariates_dict,
            rolling_window_length=self.rolling_window_length,
            stan_model=self.stan_model,
            data_generation_specification=self.data_generation_specification,
            model_hyperparameter_settings=self.model_hyperparameter_settings,
            cross_validation_settings=self.cross_validation_settings,
            sampling_parameters=self.sampling_parameters,
            validate_on_horizon=self.validate_on_horizon,
            validate_on_horizon_and_month=self.validate_on_horizon_and_month,
            is_bayesian_model=self.is_bayesian_model,
            is_baseline_model=self.is_baseline_model
        )
        return crps_score  # Optuna will try to minimize this value

    def run_optimization(self, n_trials: int = 100):
        # Define the parameter grid
        param_grid = {
            'rolling_window_length': list(range(1, 37))  # 1 to 36 inclusive
        }

        # Create the study with GridSampler
        study = optuna.create_study(sampler=GridSampler(param_grid))
        # study = optuna.create_study()
        study.optimize(self.objective, n_trials=n_trials)
        return study, study.best_params, study.best_value


# Note: Currently implementation for training on forecast horizons only, also consider the assumptions
def run_model(evaluation_year: int, model_name: str, actuals: pd.DataFrame, training_data: pd.DataFrame,
              model_run_config: dict, optuna_trials: int, validate_on_horizon: bool,
              validate_on_horizon_and_month: bool,
              is_bayesian_model: bool,
              is_baseline_model: bool,
              country_mapping: Optional[pd.DataFrame] = None,
              ) -> Tuple[Dict[str, Union[ndarray, Any]], Any, Union[DataFrame, Any]]:
    """
    Run the Bayesian model for the specified configuration.
    Assumptions:
    The forecast horizon for the first forecasted month is expected to be 3 months.
    The last month in the training data is expected to reflect that assumptions regarding the actuals data set

    Parameters:
        optuna_trials:
        evaluation_year:
        model_name:
        is_bayesian_model: indicates whether we are tuning for the Bayesian model
        is_baseline_model: indicates whether we are tuning for the baseline model
        actuals: Test data to evaluate the model
        training_data (pd.DataFrame): DataFrame containing the training data for the cross-validation process
        model_run_config (dict): Dictionary containing the model configuration.
        validate_on_horizon: Set true if the model should be cross-validated on forecast horizons only
        validate_on_horizon_and_month: Set true if the model should be cross-validated on forecast horizons and month
        is_bayesian_model: Set true if the model should be trained using Bayesian model
        is_baseline_model: Set true if the model should be trained using baseline model

    Returns:
        results (Tuple[pd.DataFrame, pd.DataFrame]): Pandas data frames containing the results of cross-validation
        and performance on actuals data
    """
    # Track the time it takes to run the model in full for one evaluation year
    start_time = time.time()
    # Ensure that one of the cross-validation identifiers is set to true
    assert validate_on_horizon is True or validate_on_horizon_and_month is True, \
        'One of the cross-validation identifiers has to be set to true'

    # Ensure that one of the model identifiers is set to true
    assert is_bayesian_model is True or is_baseline_model is True, \
        'One of the model identifiers has to be set to true'
    # transform bools into text for mlflow logging later on
    if validate_on_horizon:
        validation_approach = "forecast_horizon"
    else:
        validation_approach = "forecast_horizon_and_month"

    # Retrieve cross-validation settings from config
    cross_validation_settings = model_run_config['cross_validation_settings']
    # Retrieve K-Fold information from config
    k_fold_iterations = cross_validation_settings['k_fold_iterations']

    # Extract country_ids of countries which do have data within the evaluation period
    country_ids = sorted(actuals['country_id'].unique())
    if is_baseline_model:
        tuning_iterations = (len(country_ids) * len(actuals['month_id'].unique()) * optuna_trials)
    elif is_bayesian_model:
        tuning_iterations = (len(actuals['month_id'].unique()) * optuna_trials)

    print(f"Data Pipeline for evaluation year {evaluation_year} has been started with the following specifications: \n"
          f"model_name={model_name}, \n"
          f"validation_approach={validation_approach}, \n"
          f"k_fold_iterations={k_fold_iterations}, \n"
          f"optuna_trials={optuna_trials}, \n"
          f"no. of countries={len(country_ids)}) \n")

    if is_baseline_model:
        if validate_on_horizon:
            print(
                f"The tuning process will have a total of {tuning_iterations} iterations with 12 model trained in each "
                "iteration")
        else:
            print(
                f"The tuning process will have a total of {tuning_iterations} iterations with 1 model trained in each "
                "iteration")

    if is_bayesian_model:
        print(f"The tuning process will have a total of {tuning_iterations} iterations with 1 model trained in each "
              "iteration")

    # Calculate min and max forecast horizon corresponding to actuals data
    forecast_horizon_min, forecast_horizon_max = calculate_min_max_forecast_horizon(actuals, cross_validation_settings)

    # Create an empty dictionary to store the results of the hyperparameter tuning process
    hyperparameter_tuning_results = {}

    if is_bayesian_model is True:
        # Create empty data frame to store best results of cross validation process
        best_runs_cross_validation = pd.DataFrame(columns=["forecast_horizon", "run_id", "cross_validation_score"])

        # Unpack the rest of model_run_config dictionary
        # ToDo: More performant to do this outside the loop, but got to wait for solution to
        #       model evaluation problem (see 14.09 OneNote)
        #       With a implementation of specifying the covariates for each forecast horizon and then retrieving
        #       the corresponding ones for train and validation set, it should be feasible without the need to
        #       change the rest of the code.
        #       We would need to add dictionary to the config file for the covariates in dependence on the forecast
        #       horizon
        (stan_model, sampling_parameters, data_generation_specification, model_hyperparameter_settings,
         data_specification, descriptive_values) = unpack_model_configuration(
            model_run_config, is_bayesian_model=is_bayesian_model)
        # Retrieve covariates_dict from config
        # ToDo: Decide on one approach (data_specification bs data_generation_specification), dependent on how
        #  we solve actuals covariate data issue
        covariates_dict = data_generation_specification['covariates']

        # Compile the stan model before running the hyperparameter tuning process
        try:
            stan_model = pystan.StanModel(model_code=stan_model)
        except Exception as e:
            print(
                f'An error has occured while compiling Stan Model with Exception: {e}. The corresponding values are: \n'
                f'stan_model={stan_model}')

        forecast_tuning_times = []
        forecast_horizon_iterator = 0
        print(f"Start Hyperparameter tuning on forecast horizons {forecast_horizon_min} to {forecast_horizon_max} "
              f"for all countries simultaneously for evaluation_year {evaluation_year}.")
        # Initiate forecast_horizon iterator
        # Iterate over forecast horizons to train the model on different forecast horizons
        for forecast_horizon in range(forecast_horizon_min, forecast_horizon_max + 1):
            print(f"Start iteration on forecast horizon {forecast_horizon}...")
            start_tuning_time_forecast_horizon = time.time()
            # Retrieve target variable from config
            target_variable = data_specification['target_variable']

            # Calculate the length of the rolling window used in each fold
            rolling_window_length = calculate_rolling_window_length(training_data, k_fold_iterations, forecast_horizon)
            # Run hyperparameter tuning on forecast horizon
            # ToDo: Check if this works as intended
            # OPTIONAL: Consider logging/ storing the results of the hyperparameter tuning process (bayes)
            tuner = HyperparameterTuning(actuals=actuals,
                                         training_data=training_data,
                                         k_fold_iterations=k_fold_iterations,
                                         forecast_horizon=forecast_horizon,
                                         target_variable=target_variable,
                                         covariates_dict=covariates_dict,
                                         rolling_window_length=rolling_window_length,
                                         stan_model=stan_model,
                                         data_generation_specification=data_generation_specification,
                                         cross_validation_settings=cross_validation_settings,
                                         model_hyperparameter_settings=model_hyperparameter_settings,
                                         sampling_parameters=sampling_parameters,
                                         validate_on_horizon=validate_on_horizon,
                                         validate_on_horizon_and_month=validate_on_horizon_and_month,
                                         is_bayesian_model=is_bayesian_model)

            print(f"Start Hyperparameter tuning on forecast horizon {forecast_horizon} "
                  f"for evaluation_year {evaluation_year} with validation approach {validation_approach}...")
            start_tuning_time = time.time()
            # Run the optimization and get the study object and best parameters
            study_object, best_params, best_value = tuner.run_optimization(n_trials=optuna_trials)
            end_tuning_time = time.time() - start_tuning_time
            print(f"Hyperparameter tuning for forecast horizon {forecast_horizon} finished in "
                  f"{seconds_to_format(end_tuning_time)}")
            # Log the results of the hyperparameter tuning process
            run_id = mlflow_logging_optuna_bayesian(model_name=model_name,
                                                    evaluation_year=evaluation_year,
                                                    study_object=study_object,
                                                    sampling_parameters=sampling_parameters,
                                                    forecast_horizon=forecast_horizon,
                                                    process="cross-validation",
                                                    validation_approach=validation_approach)

            # Store the results in the dictionary
            hyperparameter_tuning_results[f"forecast_horizon_{forecast_horizon}"] = {
                "study_object": study_object,
                "best_params": best_params,
                "best_value": best_value,
                "run_id": run_id
            }
            # Note: the following is temporary and mant to be refined later on
            best_runs_cross_validation = best_runs_cross_validation.append(
                {"forecast_horizon": forecast_horizon,
                 "run_id": run_id,
                 "cross_validation_score": best_value},
                ignore_index=True)
            forecast_horizon_iterator += 1
            end_tuning_time_forecast_horizon = time.time() - start_tuning_time_forecast_horizon
            print(f" Iteration on forecast horizon {forecast_horizon} finished in"
                  f" {seconds_to_format(end_tuning_time_forecast_horizon)}")
            forecast_tuning_times.append(end_tuning_time_forecast_horizon)
            average_forecast_tuning_time = np.mean(forecast_tuning_times)
            remaining_forecast_tuning_time = ((forecast_horizon_max - forecast_horizon_min) + 1 -
                                              forecast_horizon_iterator) * average_forecast_tuning_time
            print(f"Estimated time remaining for tuning process: {seconds_to_format(remaining_forecast_tuning_time)}")

        # ToDo: Implement hyperparameter retrieving using the otuna study objects, either by taking best parameters from
        #       study object or from the exported data in mlflow log (if done), same for baseline
        print("Calculate models for actuals data...")
        # Run final model on best parameter configurations obtained from cross-validation process
        actuals_results = compute_stan_model_actual(actuals, training_data, stan_model, best_runs_cross_validation,
                                                    cross_validation_settings)
        return actuals_results, best_runs_cross_validation

    elif is_baseline_model is True:
        # Create empty data frame to store best results of cross validation process
        test_results_all_countries = pd.DataFrame(
            columns=['country_id', 'month_id', 'forecast_horizon', 'test_score', 'run_id'])

        # OPTIONAL: Improve hard coded part
        max_quantiles = 99
        columns_contribution_results = ["month_id", "country_id", "draw"]

        # Pre-allocate columns for maximum number of quantiles
        for i in range(1, max_quantiles + 1):
            columns_contribution_results.append(f"sample_{i}")

        competition_contribution_results = pd.DataFrame(columns=columns_contribution_results)
        # Unpack the rest of model_run_config dictionary
        (model_hyperparameter_settings, data_specification, descriptive_values) = unpack_model_configuration(
            model_run_config, is_baseline_model=is_baseline_model)

        # ToDo: Remove
        # Filter country_ids on countries 133, 124, 220 for testing purposes
        # country_ids = [246, 133, 1]
        # Take last 20 entries of list country_ids
        # country_ids = country_ids[-20:]

        # Retrieve target variable from config
        target_variable = data_specification['target_variable']

        # Note: originally loop over horizons and then over countries, which is more performant (less joins, less appends),
        #       however this it's a trade-off between performance and readability
        # Iterate over each country
        country_tuning_times = []
        # Iterator to track the number of countries for which the tuning process has been finished
        country_iterator = 0
        print(f"Start Hyperparameter tuning on {len(country_ids)} countries for evaluation_year {evaluation_year}.")
        for country_id in country_ids:
            start_tuning_time_country = time.time()
            # Create empty data frame to store best results of cross validation process for a single country
            best_runs_cross_validation_country = pd.DataFrame(
                columns=["forecast_horizon", "country_id", "run_id", "cross_validation_score"])
            # Select data for country
            actuals_country_specific = actuals[actuals['country_id'] == country_id]
            training_data_country_specific = training_data[training_data['country_id'] == country_id]

            # Iterate over forecast horizons to train the model on different forecast horizons
            for forecast_horizon in range(forecast_horizon_min, forecast_horizon_max + 1):
                # toDo: Check if we need this for the bayesian model as well
                # Respective to the actuals data, we want to determine how many months we need to perform out
                # cross-validation process. This is calculated here in a first step.
                # Idea: The first cv-set is at most K times the length of the actuals data ahead of the first
                #       actuals month. In our case K years.
                #       For this set we would additionally need the current forecast_horizon and the maximum
                #       rolling_window_length for training.
                maximum_months_needed_before_eval_year = (
                        model_hyperparameter_settings["rolling_window_length"]["range"][1] +
                        forecast_horizon +
                        k_fold_iterations * len(actuals_country_specific))

                # ToDo: adjust for true future prediction of 2024
                #       Probably the easiest way is to adjust the code so each the training input goes until Oct before
                #       the evaluation year. However we should also consider the change of competition rules in the
                #       forecasting period
                # This differentiation pays respect to the fact that Nov20 and Dec20 are not available
                if evaluation_year == 2021:
                    months_available_before_eval_year = len(training_data_country_specific)
                else:
                    months_available_before_eval_year = (
                            actuals['month_id'].min() - training_data_country_specific['month_id'].min())

                # Some countries may not have sufficient data for the intended K Fold iterations which is why they have
                # to be adjusted in those cases
                if months_available_before_eval_year < maximum_months_needed_before_eval_year:
                    # Determine how many k fold iterations are possible with the available data dependent on the
                    # true future evaluation set
                    # I want to explain it as follows:
                    # if we are in the year 2018, 2019, 2020 this will deduct the maximum rolling window length and the
                    # current forecast horizon from the available months before the evaluation year and then return the
                    # number of full years which fit into the remaining months. This will be the years from Jan till Dec
                    # before the evaluation year used as cv sets
                    # if we are in the year 2021 we also calculate the remaining full years, however the data for the
                    # cross validation sets ranges from Nov(t) till Oct(t+K-1). For further context compare the
                    # for get_validation_months_for_fold function
                    months_available_for_cv_sets = (months_available_before_eval_year -
                                                    model_hyperparameter_settings["rolling_window_length"]["range"][1] -
                                                    forecast_horizon)
                    # OPTIONAL: Modification for forecast_horizon as we do not need full years for 2021 here. we could
                    #           one iteration to the one calculated below, if the remainder of the division is 10 or 11
                    k_fold_iterations = months_available_for_cv_sets // len(actuals_country_specific)

                # OPTIONAL: Consider logging/ storing the results of the hyperparameter tuning process (baseline)
                # Run hyperparameter tuning on forecast horizon for each country
                tuner = HyperparameterTuning(actuals=actuals_country_specific,
                                             training_data=training_data_country_specific,
                                             k_fold_iterations=k_fold_iterations,
                                             forecast_horizon=forecast_horizon,
                                             target_variable=target_variable,
                                             model_hyperparameter_settings=model_hyperparameter_settings,
                                             validate_on_horizon=validate_on_horizon,
                                             validate_on_horizon_and_month=validate_on_horizon_and_month,
                                             is_baseline_model=is_baseline_model)

                print(
                    f"Start Hyperparameter tuning on country {country_id} and forecast horizon {forecast_horizon} "
                    f"for evaluation_year {evaluation_year} with validation approach {validation_approach}...")
                start_tuning_time = time.time()
                # Run the optimization and get the study object and best parameters
                study_object, best_params, best_value = tuner.run_optimization(n_trials=optuna_trials)
                end_tuning_time = time.time() - start_tuning_time
                print(
                    f"Hyperparameter tuning for country {country_id} and forecast horizon {forecast_horizon} "
                    f"and evaluation_year {evaluation_year} finished in {seconds_to_format(end_tuning_time)}.")

                run_id = mlflow_logging_optuna_baseline(model_name=model_name,
                                                        evaluation_year=evaluation_year,
                                                        country_id=country_id,
                                                        country_mapping=country_mapping,
                                                        k_fold_iterations=k_fold_iterations,
                                                        study_object=study_object,
                                                        forecast_horizon=forecast_horizon,
                                                        process="cross-validation",
                                                        validation_approach=validation_approach)

                # Store the results in the dictionary
                hyperparameter_tuning_results[f"forecast_horizon_{forecast_horizon}"] = {
                    "study_object": study_object,
                    "best_params": best_params,
                    "best_value": best_value,
                    "country_id": country_id,
                    "run_id": run_id,
                    "k_fold_iterations": k_fold_iterations
                }
                # Reset k_fold_iterations to original value
                k_fold_iterations = cross_validation_settings['k_fold_iterations']
                # Note: the following is temporary and mant to be refined later on
                best_runs_cross_validation_country = best_runs_cross_validation_country.append(
                    {"forecast_horizon": forecast_horizon,
                     "run_id": run_id,
                     "cross_validation_score": best_value,
                     "country_id": country_id
                     },
                    ignore_index=True)
                end_tuning_time_country = time.time() - start_tuning_time_country

            # Run final model on best parameter configurations obtained from cross-validation process
            test_result_country, competition_contribution_results_country_specific = compute_baseline_model_actual(
                actuals_country_specific, training_data_country_specific, best_runs_cross_validation_country,
                model_name, country_id, columns_contribution_results=columns_contribution_results)
            # Append the results for the country to the results for all countries
            test_results_all_countries = test_results_all_countries.append(test_result_country, ignore_index=True)
            competition_contribution_results = competition_contribution_results.append(
                competition_contribution_results_country_specific, ignore_index=True)
            end_tuning_time_country = time.time() - start_tuning_time_country
            print(f'Finished tuning for country {country_id} and calculating on actuals in '
                  f'{seconds_to_format(end_tuning_time_country)}')
            country_tuning_times.append(end_tuning_time_country)
            country_iterator += 1
            average_country_tuning_time = np.mean(country_tuning_times)
            # Calculate estimated remaining tuning time based on average tuning time per country and remaining countries
            # in for loop
            remaining_tuning_time = (len(country_ids) - country_iterator) * average_country_tuning_time
            print(f'Estimated remaining tuning time: {seconds_to_format(remaining_tuning_time)} '
                  f'for evaluation year {evaluation_year}')

        # Group by 'forecast_horizon' and 'month_id' and calculate the mean and std of 'cross_validation_score'
        test_results_global = test_results_all_countries.groupby(['forecast_horizon', 'month_id'])[
            'test_score'].agg(['mean', 'std']).reset_index()

        # Rename the columns
        test_results_global = test_results_global.rename(columns={
            'mean': 'global_test_score',
            'std': 'global_test_score_std_between_countries'
        })

        # Calculate the average of the global_test_score over the validation months
        average_global_test_score = np.mean(test_results_global['global_test_score'])

        # Calculate the standard deviation of the global_test_score across the validation months
        std_global_test_score = np.std(test_results_global['global_test_score'])

        # Create a dictionary to store the results
        test_results_global_dict = {
            "average_global_test_score": average_global_test_score,
            "std_global_test_score": std_global_test_score,
            "test_results_global": test_results_global
        }
        competition_contribution_results['month_id'] = competition_contribution_results['month_id'].astype(int)
        competition_contribution_results['country_id'] = competition_contribution_results['country_id'].astype(int)
        competition_contribution_results['draw'] = competition_contribution_results['draw'].astype(int)
        run_time_one_evaluation_year = time.time() - start_time
        ml_flow_logging_actual_baseline(model_name=model_name,
                                        evaluation_year=evaluation_year,
                                        process="model-testing",
                                        validation_approach=validation_approach,
                                        test_results_global_dict=test_results_global_dict,
                                        test_results_all_countries=test_results_all_countries,
                                        competition_contribution_results=competition_contribution_results,
                                        run_time=run_time_one_evaluation_year)
        # ToDo: Partly redundant return
        return test_results_global_dict, test_results_global, test_results_all_countries, competition_contribution_results


def calculate_min_max_forecast_horizon(actuals: pd.DataFrame, cross_validation_settings: dict) -> Tuple[int, int]:
    """
    Calculate the minimum and maximum forecast horizon for the specified actuals data set.
    Args:
        actuals: test data set after cross-validation process
        model_run_config: dictionary containing the model configuration

    Returns:
        forecast_horizon_min (int): forecast horizon for first month in test set
        forecast_horizon_max (int): forecast horizon for last month in test set


    """
    # Retrieve forecast horizon for first month from config
    forecast_horizon_min = cross_validation_settings['forecast_horizon']
    # Calculate months to be forecasted
    no_forecasted_months = len(actuals['month_id'].unique())
    # Calculate the maximum forecast horizon, forecast horizon for last month in test set
    forecast_horizon_max = forecast_horizon_min + no_forecasted_months - 1

    return forecast_horizon_min, forecast_horizon_max


def calculate_rolling_window_length(training_data: pd.DataFrame, k: int, forecast_horizon: int) -> int:
    """
    Calculate the length of the rolling window for the specified number of folds. The length of the rolling window is
    dependent on the number of folds and corresponds to the maximum number of months that can be used for validating
    first cross-validation set. This is only needed for the Bayesian model.
    Args:
        forecast_horizon:
        training_data: DataFrame containing the training data for the cross-validation process
        k: indicates which fold iteration we are in
    Returns:

    """
    # Retrieve last month in training data
    last_observation_month = get_last_observation_month(training_data)

    # ToDo: Check abbreviation for Quick fix for missing Nov, Dec for last validation set, remove if other approach
    #       taken
    # Shift the last observation date by K years to determine last training month of first k fold
    last_training_month = last_observation_month - 12 * k

    # ToDo: Implement modularity for rolling window if lag features are used  (do not add forecast horizon if not)
    # Find the first observation date in the training data and adjust by forecast_horizon as lag features not available
    # for first observations
    first_observation_month = training_data['month_id'].min() + forecast_horizon

    # Calculate the number of months comprised by first observed month and last training month
    rolling_window_length = last_training_month - first_observation_month + 1

    return rolling_window_length


def get_last_observation_month(training_data: pd.DataFrame) -> int:
    """
    Get the last observation date in the training data.
    Assumption: This month is the October of the year previous to the test data set.
    Args:
        training_data: DataFrame containing the training data for the cross-validation process

    Returns: last_observation_month (int): last observed month in training data

    """
    # Sort the training data by 'datetime' column
    training_data.sort_values(by='month_id', inplace=True)
    # Find the last observation date in the training data
    last_observation_month = training_data['month_id'].max()

    return last_observation_month


def unpack_model_configuration(model_run_config: dict, is_bayesian_model: Optional[bool] = None,
                               is_baseline_model: Optional[bool] = None):
    """
    Unpack the model configuration dictionary.
    Args:
        model_run_config: dictionary containing the model configuration
        is_baseline_model: indicates which dictionary to unpack for baseline model
        is_bayesian_model: indicates which dictionary to unpack for bayesian model

    Returns: unpacked_config (tuple): tuple containing the unpacked model configuration

    """
    # Unpack the model_run_config dictionary
    stan_model = model_run_config.get("stan_model_code", None)
    data_generation_specification = model_run_config.get("data_generation_specification", None)
    model_hyperparameter_settings = model_run_config.get("model_hyperparameter_settings", None)
    sampling_parameters = model_run_config.get("sampling_parameters", None)
    data_specification = model_run_config["data_specification"]
    descriptive_values = model_run_config["descriptive_values"]

    if is_bayesian_model is True:
        return (stan_model, sampling_parameters, data_generation_specification, model_hyperparameter_settings,
                data_specification, descriptive_values)
    elif is_baseline_model is True:
        return (model_hyperparameter_settings, data_specification, descriptive_values)


def run_hyperparameter_tuning(trial, actuals: pd.DataFrame, training_data: pd.DataFrame, k_fold_iterations: int,
                              forecast_horizon: int, target_variable: str, covariates_dict: dict,
                              rolling_window_length: Optional[int] = None,
                              stan_model: Optional[pystan.StanModel] = None,
                              data_generation_specification: Optional[dict] = None,
                              model_hyperparameter_settings: Optional[dict] = None,
                              cross_validation_settings: Optional[dict] = None,
                              sampling_parameters: Optional[dict] = None, validate_on_horizon: Optional[bool] = None,
                              validate_on_horizon_and_month: Optional[bool] = None,
                              is_bayesian_model: Optional[bool] = None, is_baseline_model: Optional[bool] = None,
                              ) -> float:
    """
    Args:
        cross_validation_settings:
        data_generation_specification:
        covariates_dict:
        trial: optuna trial object
        actuals: test set after cross-validation process
        training_data: DataFrame containing the training data for the cross-validation process
                       (filtered on a country for baseline model)
        k_fold_iterations: indicates the number of folds to be used in the cross-validation process
        forecast_horizon: forecast horizon in months for which we tune the hyperparameters
        target_variable:
        rolling_window_length: Optional parameter as only needed for Bayesian model.
        stan_model:  Optional parameter as only needed for Bayesian model.
        model_hyperparameter_settings: dict containing information about the hyperparameters to be tuned
        sampling_parameters: Optional parameter as only needed for Bayesian model.
        validate_on_horizon: indicates whether the model should be cross-validated on forecast horizons only
        validate_on_horizon_and_month: indicates whether the model should be cross-validated on forecast horizons and
                                       month
        is_bayesian_model: indicates whether we are tuning for the Bayesian model
        is_baseline_model: indicates whether we are tuning for the baseline model

    Returns: average_crps_k_fold_iterations (float): average CRPS score across all folds

    """
    # ToDo: Remove this empty data frame if not needed
    # Create empty DataFrame to store results of cross-validation process for a single forecast horizon
    crps_scores_cross_validation_single_h = pd.DataFrame(columns=['no_k_folds', 'forecast_horizon', 'average_crps',
                                                                  'std_crps', 'run_id'])
    # Generate empty list to store the CRPS scores of single k_fold validation sets
    crps_scores_k_fold_iterations = []
    # Iterate over the number of folds
    for k in range(1, k_fold_iterations + 1):
        # Retrieve the validation set for the current fold
        validation_set_months = get_validation_months_for_fold(actuals, training_data, k, k_fold_iterations, forecast_horizon,
                                                               validate_on_horizon, validate_on_horizon_and_month,
                                                               is_baseline_model=is_baseline_model,
                                                               is_bayesian_model=is_bayesian_model)
        # Generate empty list to store the CRPS scores for the validation set
        crps_scores_val_set = []
        # Note:
        #  For performance gain, we could consider training the model once on training data till
        #  validation_set.min() - forecast horizon and then evaluating all validation months together (instead of
        #  training the model for each validation month separately with up to 11 months additional data). Although
        #  this setting is not reproducing the actual forecasting task (how would we have the input features for all
        #  validation months at the forecast origin), the training on all months of the year is still justified as we
        #  may gain generalization patterns which could get lost compared to limiting us on the specific months
        #  corresponding to the forecast horizon. Training on months till validation_set.max() - forecast horizon
        #  cannot be justified as this would cause target leakage.
        #  In the following we attempt to implement the performance gain approach.
        # Iterate over months to be forecasted

        if is_bayesian_model is True:
            # Get covariates corresponding to the current forecast horizon from data_generation_specification
            covariates = get_covarariates_for_forecast_horizon(covariates_dict, forecast_horizon)
            # Select the columns to be used for training the model
            training_data = training_data[['month_id'] + [target_variable] + covariates]
            # Determine first month in validation set, this will be used to determine a training set, that does not
            # contain any data we wouldn't have at the forecast origin normally
            first_validation_month = validation_set_months['month_id'].min()
            # Retrieve the training set corresponding to current validation month
            train_set = get_train_set_for_fold(training_data, first_validation_month, forecast_horizon,
                                               rolling_window_length,
                                               target_variable, covariates, is_bayesian_model=is_bayesian_model)
            # Get the validation data corresponding to the set of validation months (validation set)
            validation_set = training_data[training_data['month_id'].isin(validation_set_months['month_id'])]
            crps_average_over_val_months = compute_stan_model(trial, stan_model, data_generation_specification,
                                                              cross_validation_settings, model_hyperparameter_settings,
                                                              sampling_parameters, train_set, validation_set,
                                                              target_variable,
                                                              covariates)
            # Append the CRPS score to the list
            crps_scores_val_set.append(crps_average_over_val_months)

        elif is_baseline_model is True:
            training_data = training_data[['month_id'] + [target_variable]]
            for val_month in validation_set_months['month_id']:
                # Train model on training set and cross-validate on validation month using the specified model
                # Note: as for the baseline the length of the train set is a hyperparameter itself, it its specified
                #       within the compute_baseline_model function
                crps_val_month, _ = compute_baseline_model(trial, training_data, val_month, model_hyperparameter_settings,
                                                        forecast_horizon, target_variable)
                # Append the CRPS score to the list
                crps_scores_val_set.append(crps_val_month)
        # crps_scores_val_set
        # (i) baseline: either 1 month or 12 months
        # (ii) bayesian: either 1 month or average over 12 months
        # Calculate the average CRPS score for the validation set
        # Note: This averaging is redundant in some cases, but does no harm
        average_crps_val_set = np.mean(crps_scores_val_set)
        # Append the average CRPS score of one fold to the k_fold_iterations list
        crps_scores_k_fold_iterations.append(average_crps_val_set)

    # Calculate the average and standard deviation of the CRPS scores across all folds
    average_crps_k_fold_iterations = np.mean(crps_scores_k_fold_iterations)

    return average_crps_k_fold_iterations


# ToDo: Inspect if it's sufficient to return the months as set because we obtain the actual value from training data in
#       cross validation process
def get_validation_months_for_fold(actuals: pd.DataFrame, training_data: pd.DataFrame, k: int, K: int,
                                   forecast_horizon: int,
                                   validate_on_horizon: Optional[bool] = None,
                                   validate_on_horizon_and_month: Optional[bool] = None,
                                   is_baseline_model: Optional[bool] = None,
                                   is_bayesian_model: Optional[bool] = None) -> pd.DataFrame:
    """
    Get the validation set for the specified fold.
    Note: If we validate on forecast horizon and month this function will return a single month only.
    Args:
        K: Total intended number of folds
        forecast_horizon: forecast horizon in months
        validate_on_horizon: indicates whether the model should be cross-validated on forecast horizons only
        validate_on_horizon_and_month: indicates whether the model should be cross-validated on forecast horizons and
        actuals: test set after cross-validation process
        training_data: DataFrame containing the training data for the cross-validation process
        k:

    Returns:

    """
    # Note: We encountered the problem that for evaluation year t, Nov + Dec t-1 can't be used for cross-validation
    #       as we don't have data for these months.
    #       Quick fix: We move the cross-validation sets one year back, in the following incorporate by "-12".
    #                  This goes at the cost of loosing Jan - Oct t-1 for cross-validation.
    #       Sophisticated fix: We could define the validation sets as Nov - Oct.
    #                          For validate_on_horizon this would be achieved by substracting '-2' from first and last
    #                          validation month. For validate_on_horizon_and_month the determination of the validation
    #                          month based in first_validation_month would be need to match the forecast horizons
    #                          [3, 4, ..., 14] to something like first_validation_month + [2, 3,..., 11, 0, 1].
    #                          Effects on the rest of the pipeline should be considered.
    #                          Keeping of time order should be ensured (e.g. sort validation set)
    # Derive validation sets from actuals data
    # First month in validation set
    first_validation_month = actuals['month_id'].min() - 12 * k
    months_available_for_validation = training_data["month_id"].max() - (first_validation_month - 1)

    if validate_on_horizon is True:
        if months_available_for_validation < 12:
            last_validation_month = first_validation_month - 1 + months_available_for_validation
        else:
            # Last month in validation set
            last_validation_month = actuals['month_id'].max() - 12 * k
        # Get validation set
        validation_set = training_data[training_data['month_id'].between(first_validation_month, last_validation_month)]

    # ToDo: Inspect if works as intended
    if validate_on_horizon_and_month is True:
        # Calculate the validation month in accordance to forecast horizon
        validation_month = first_validation_month + forecast_horizon - 3
        # Check if there is a corresponding observation in training_data
        if validation_month not in training_data['month_id'].values:
            validation_month_adjusted = validation_month - 12 * K
            validation_set = training_data[training_data['month_id'] == validation_month_adjusted]
        else:
            # Get validation set
            validation_set = training_data[training_data['month_id'] == validation_month]

    if is_bayesian_model is True:
        return validation_set[['month_id']]
    elif is_baseline_model is True:
        return validation_set[['month_id']]


def get_train_set_for_fold(training_data: pd.DataFrame, validation_month: int, forecast_horizon: int,
                           rolling_window_length: int, target_variable: str,
                           covariates: Optional[list] = None,
                           is_bayesian_model: Optional[bool] = None,
                           is_baseline_model: Optional[bool] = None) -> pd.DataFrame:
    """
    Returns the training set corresponding to a specific validation month and forecast horizon with length of the
    rolling window. This function is only used for the Bayesian model.
    Args:
        is_baseline_model:
        is_bayesian_model:
        target_variable:
        covariates:
        training_data: DataFrame containing the training data for the cross-validation process
        validation_month:
        forecast_horizon: forecast horizon in months
        rolling_window_length: length of the rollin window in months for training data in each fold of cross-validation

    Returns:

    """
    last_training_month = validation_month - forecast_horizon
    first_training_month = last_training_month - rolling_window_length + 1
    train_set = training_data[training_data['month_id'].between(first_training_month, last_training_month)]

    # ToDo: Check if its suitable to drop the month_id column
    if is_bayesian_model is True:
        return train_set[[target_variable] + covariates]
    elif is_baseline_model is True:
        return train_set[[target_variable]]


def compute_stan_model(trial: optuna.Trial, stan_model: pystan.StanModel, data_generation_specification: dict,
                       cross_validation_settings: dict, model_hyperparameter_settings: dict, sampling_parameters: dict,
                       train_set: pd.DataFrame, validation_set: pd.DataFrame, target_variable: str,
                       covariates: list) -> float:
    """
    Compute the Bayesian model with the hyperparamters delivered by Optuna.
    Args:
        cross_validation_settings:
        target_variable:
        covariates:
        data_generation_specification:
        trial:
        stan_model:
        model_hyperparameter_settings:
        sampling_parameters:
        train_set:
        validation_set: validation set for the current validation month with covariates and target variable

    Returns:

    """
    stan_data = {}
    # Get the target data and derive the needed stats such as length of the data
    dataY = train_set[target_variable].values
    dataY_eval = validation_set[target_variable].values
    stan_data["Y"] = dataY.astype(int)
    stan_data["Y_eval"] = dataY_eval.astype(int)
    stan_data["no_data"] = len(dataY)
    stan_data["no_data_eval"] = len(dataY_eval)
    # Get the static parameters of the configuration dictionary and add them to the stan_data dictionary
    for key, value in data_generation_specification.items():
        if key == "random_walk_order":
            for key2, value2 in value.items():
                stan_data[key2] = value2
        else:
            stan_data[key] = value

    # Get the dynamic parameters of the configuration dictionary, initialize them as hyperparameters and add them to
    # the stan_data dictionary
    for key, value in model_hyperparameter_settings.items():
        if key == "no_interior_knots":
            for key2, value2 in value.items():
                param_type = value2['type']
                param_range = value2['range']

                if param_type == 'int':
                    stan_data[key2] = trial.suggest_int(key2, param_range[0], param_range[1])
                elif param_type == 'float':
                    stan_data[key2] = trial.suggest_float(key2, param_range[0], param_range[1])
        else:
            param_type = value['type']
            param_range = value['range']

            if param_type == 'int':
                stan_data[key] = trial.suggest_int(key, param_range[0], param_range[1])
            elif param_type == 'float':
                stan_data[key] = trial.suggest_float(key, param_range[0], param_range[1])

    # FixMe: We attempt to use covariate data at the time of the evaluation month which is not available in the true
    #  future
    # Specify covariate data passed to stan model and parameters subject to other hyperparameters
    covariate_iterator = 1
    for covariate in covariates:
        stan_data[f"covariate_data_X{covariate_iterator}"] = train_set[covariate].values
        stan_data[f"covariate_data_X{covariate_iterator}_eval"] = validation_set[covariate].values
        stan_data[f"no_basis_X{covariate_iterator}"] = (
                stan_data[f"no_interior_knots_X{covariate_iterator}"] + stan_data["spline_degree"])
        covariate_iterator += 1

    # Fit the stan model
    posterior_predicitve = cross_validation_settings['posterior_predictive']
    start_time = time.time()
    stan_model_fit = stan_model.sampling(data=stan_data, **sampling_parameters)
    sampling_time = time.time() - start_time
    idata = az.from_pystan(posterior=stan_model_fit, posterior_predictive=posterior_predicitve)
    # ToDo: Adjust calculate_crps function to input array
    # ToDo: Check if we want to calculate the scores for the training set as well
    crps_train, crps_average_train = calculate_crps(idata, dataY, posterior_predictive=posterior_predicitve[0])
    crps_eval, crps_average_eval = calculate_crps(idata, dataY_eval, posterior_predictive=posterior_predicitve[1])

    return crps_average_eval


# ToDo: Create features for this dynamic covariate
def get_covarariates_for_forecast_horizon(covariates_dict: dict, forecast_horizon: int) -> list:
    """
    Get the covariates corresponding to the specified forecast horizon.
    Args:
        data_generation_specification: dictionary containing the data generation specification
        forecast_horizon: forecast horizon in months

    Returns: covariates (list): list of covariates for model corresponding to the specified forecast horizon

    """
    general_covariates = covariates_dict['general_covariates']
    horizon_specific_covariates = covariates_dict['horizon_specific_covariates'][f'forecast_horizon_{forecast_horizon}']
    covariates = general_covariates + horizon_specific_covariates

    return covariates


# ToDo: ! Change if not fixed quantiles anymore!
def compute_baseline_model(trial, training_data: pd.DataFrame, validation_month: int,
                           model_hyperparameter_settings: dict, forecast_horizon: int, target_variable: str,
                           is_baseline_model=True, use_fixed_quantiles=True) -> float:
    """
    Compute the baseline model for the specified configuration.
    Args:
        is_baseline_model:
        target_variable:
        trial:
        forecast_horizon:
        training_data: DataFrame containing the training data for the cross-validation process
        model_hyperparameter_settings: dictionary with hyperparameter settings for baseline model
        validation_month: month_id of validation month

    Returns:

    """
    # Get the hyperparameters of the configuration dictionary
    rolling_window_length = trial.suggest_int('rolling_window_length',
                                              model_hyperparameter_settings['rolling_window_length']['range'][0],
                                              model_hyperparameter_settings['rolling_window_length']['range'][1])
    if not use_fixed_quantiles:
        n_quantiles = trial.suggest_int('n_quantiles',
                                        model_hyperparameter_settings['quantiles']['n_quantiles']['range'][0],
                                        model_hyperparameter_settings['quantiles']['n_quantiles']['range'][1])
        quantiles = []
        # ToDo: Adjust this iteration so the same quantile does not appear twice in the selected quantiles
        for i in range(n_quantiles):
            quantile = trial.suggest_float(f'quantile_{i + 1}',
                                           model_hyperparameter_settings['quantiles']['quantile_value']['range'][0],
                                           model_hyperparameter_settings['quantiles']['quantile_value']['range'][1],
                                           step=model_hyperparameter_settings['quantiles']['quantile_value']['step'])
            quantiles.append(quantile)
        # Sort the quantiles to ensure they are in ascending order
        quantiles = sorted(quantiles)
    else:
        # Use fixed quantiles from 0.01 to 0.99 in 0.01 steps
        quantiles = [i / 100 for i in range(1, 100)]
    # Compute the training set for the current validation month
    train_set = get_train_set_for_fold(training_data=training_data,
                                       validation_month=validation_month,
                                       forecast_horizon=forecast_horizon,
                                       rolling_window_length=rolling_window_length,
                                       target_variable=target_variable,
                                       is_baseline_model=is_baseline_model)
    observation = training_data[training_data['month_id'] == validation_month][target_variable].values[0]
    crps, quantiles_calculated = compute_baseline_crps(train_set, observation, quantiles)

    return crps, quantiles_calculated


def compute_baseline_crps(train_set: pd.DataFrame, observation: float, quantiles: list) -> Tuple[float, DataFrame]:
    """
    Performs forecast for the specific model settings and returns the CRPS value of the prediction for forecast_month

    :param observation: Actual value of forecasting month
    :param train_set: training set respective to forecast horizon and training months
    :param quantiles: quantiles we want to compute as forecast sample
    :return: CRPS value of the prediction for forecasting month
    """

    # Compute quantiles of training data for the specified quantiles
    quantiles = compute_quantiles(train_set, quantiles)

    # Compute forecast samples based on quantiles computed from training data
    crps = compute_crps_baseline(observation=observation, quantiles=quantiles)

    return crps, quantiles


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


def compute_crps_baseline(observation: float, quantiles: pd.DataFrame) -> float:
    """
    Compute CRPS score based on quantiles as forecasting sample for forecasting month
    :param observation: Actual value of forecasting month
    :param quantiles: Quantiles with corresponding quantile values
    :return: CRPS score forecasting month
    """
    # Compute CRPS score based on quantile sample for forecast_month
    quantile_values = np.array(quantiles['quantile_values'])
    crps, _, _ = pscore(quantile_values, observation).compute()

    return crps


def get_best_runs(crps_scores_cross_validation: pd.DataFrame) -> pd.DataFrame:
    """
    Gets the best runs of our cross-validation process for each forecast horizon to identify the models to be used for
    the test set.
    Args:
        crps_scores_cross_validation: dataframe containing the results of each model iteration
         (differentiated by different parameters) of the cross-validation process for all forecast horizons

    Returns: best_runs_crps (pd.DataFrame): dataframe containing the best runs of our cross-validation process for each
                                            forecast horizon

    """
    best_runs_crps = crps_scores_cross_validation.groupby(['forecast_horizon'])['average_crps'].min()

    return best_runs_crps


def compute_stan_model_actual(actuals: pd.DataFrame, training_data: pd.DataFrame, stan_model: pystan.StanModel,
                              best_runs_cross_validation: pd.DataFrame, cross_validation_settings: dict) -> dict:
    """

    Args:
        cross_validation_settings:
        actuals: test set after cross-validation process
        training_data: DataFrame containing the training data for the cross-validation process
        stan_model: stan model code as string
        best_runs_cross_validation: dataframe containing the best runs of our cross-validation process for each
                                    forecast horizon

    Returns:

    """
    actuals_run_specification, train_set = prepare_actual_run_bayes(actuals, training_data, best_runs_cross_validation)
    for index, row in actuals_run_specification.iterrows():
        forecast_horizon = row['forecast_horizon']
        run_id = row['run_id']
        val_month = row['month_id']
        actual_value = row['ged_sb']
        # Run stan model with best parameter settings of forecast horizon
        # Construct the file path the parameter settings are stored in
        file_path = f"mlruns/1/{run_id}/model{run_id}_parameter_settings.json"

        stan_data, sampling_parameters, data_generation_specification = (
            read_test_model_parameters(file_path, is_bayesian_model=True))
        start_time = time.time()
        target_variable = data_generation_specification['target_variable']
        # Get the target data and derive the needed stats such as length of the data
        covariates = data_generation_specification['covariates']
        # Specify covariate data passed to stan model and parameters subject to other hyperparameters
        covariate_iterator = 1
        for covariate in covariates:
            stan_data[f"covariate_data_X{covariate_iterator}"] = train_set[covariate].values
            # ToDo: ! Here we run into the expected data definition problem !
            stan_data[f"covariate_data_X{covariate_iterator}_eval"] = validation_month[covariate].values
            stan_data[f"no_basis_X{covariate_iterator}"] = (
                    stan_data[f"no_interior_knots_X{covariate_iterator}"] + stan_data["spline_degree"])
            covariate_iterator += 1
        dataY = train_set[target_variable].values
        dataY_eval = actual_value
        stan_data["Y"] = dataY
        stan_data["Y_eval"] = dataY_eval
        stan_data["no_data"] = len(dataY)
        stan_data["no_data_eval"] = len(dataY_eval)
        # Get the static parameters of the configuration dictionary and add them to the stan_data dictionary
        stan_data["spline_degree"] = data_generation_specification['spline_degree']
        stan_model_fit = stan_model.sampling(data=stan_data, **sampling_parameters)
        posterior_predicitve = cross_validation_settings['posterior_predictive']
        idata_stan_model = az.from_pystan(posterior=stan_model_fit,
                                          posterior_predictive=posterior_predicitve)
        # ToDo: Possibly this has to be adjusted
        crps_train, crps_average_train = calculate_crps(idata_stan_model, dataY,
                                                        posterior_predictive=posterior_predicitve[0])
        crps_eval, crps_average_eval = calculate_crps(idata_stan_model, dataY_eval,
                                                      posterior_predictive=posterior_predicitve[1])
        # ToDo: Implement logging of results
        # Append the CRPS score to the column 'crps_score' in actuals_run_specification
        actuals_run_specification.loc[
            (actuals_run_specification['month_id'] == val_month), 'crps_score'] = crps_average_eval

    crps_results = actuals_run_specification
    # Calculate the average CRPS score across all validation months
    average_crps = np.mean(crps_results['crps_score'])
    # Calculate the standard deviation of the CRPS scores across all validation months
    std_crps = np.std(crps_results['crps_score'])
    actual_results = {
        "average_crps": average_crps,
        "std_crps": std_crps,
        "crps_results": crps_results
    }

    return actual_results


def prepare_actual_run_bayes(actuals: pd.DataFrame, training_data: pd.DataFrame,
                             best_runs_cross_validation: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    """
    Builds the actuals_run_specification dataframe which contains best model specification for each forecast horizon
    and the corresponding actuals data.
    Args:
        actuals: test set after cross-validation process
        training_data: DataFrame containing the training data for the cross-validation process
        best_runs_cross_validation: dataframe containing the best runs of our cross-validation process for each
                                    forecast horizon

    Returns:

    """
    # Sort best runs dataframe by forecast horizon
    best_runs_cross_validation = (
        best_runs_cross_validation.sort_values(by='forecast_horizon').reset_index(drop=True))

    # Sort actuals dataframe by month_id for sanity
    actuals = actuals.sort_values(by='month_id').reset_index(drop=True)
    # ToDo: Check functionality
    # Add a forecast_horizon column to actuals dataframe ranging from 3 to 14
    actuals['forecast_horizon'] = actuals['month_id'] - actuals['month_id'].min() + 3

    # Join best_runs_cross_validation and actuals on forecast_horizon
    actuals_run_specification = pd.merge(best_runs_cross_validation, actuals, on='forecast_horizon')
    # Create empty columns 'crps_score' in actuals_run_specification to store crps results afterward
    actuals_run_specification['crps_score'] = np.nan

    # Determine last month of training data
    last_training_month = (
            actuals_run_specification['month_id'].min() - actuals_run_specification['forecast_horizon'].min())
    # Determine train data set and return it if bayesian model
    train_set = training_data[training_data['month_id'] <= last_training_month]

    return actuals_run_specification, train_set


def compute_baseline_model_actual(actuals: pd.DataFrame, training_data: pd.DataFrame,
                                  best_runs_cross_validation: pd.DataFrame, model_name: str, country_id: int, columns_contribution_results: list,
                                  use_fixed_quantiles = True) -> (
        float, float, pd.DataFrame):
    """
    Runs the baseline model for the best configurations of the cross-validation process on the actuals data for each
    forecast horizon of a country.
    Args:
        actuals: test set after cross-validation process
        training_data: DataFrame containing the training data for the cross-validation process
        best_runs_cross_validation: dataframe containing the best runs of our cross-validation process for each
                                    forecast horizon of one country.

    Returns:
        average_crps (float): average CRPS score across all validation months and countries (global)
        std_crps (float): standard deviation of the CRPS scores within validation months scores of global score
        crps_results_global (pd.DataFrame): dataframe containing the average crps score for each validation month over
                                            all countries
        crps_results_country_specific (pd.DataFrame): dataframe containing the crps score for each country and
                                                      validation month

    """
    competition_contribution_results_country_specific = pd.DataFrame(columns=columns_contribution_results)
    actuals_run_specification = prepare_actual_run_baseline(actuals, best_runs_cross_validation)
    # Determine last month of training data (same for every validation month)
    last_training_month = (
            actuals_run_specification['month_id'].min() - actuals_run_specification['forecast_horizon'].min())
    # Run baseline model with the best parameter settings for each forecast horizon
    for index, row in actuals_run_specification.iterrows():
        # Note: we don't need tp access the forecast horizon explicitly as it is given by the validation month
        #       implicitly
        run_id = row['run_id']
        val_month = row['month_id']
        actual_value = row['ged_sb']

        # Construct the file path the parameter settings are stored in
        # ToDo: Check that baseline projects are stored under 2
        file_path = f"mlruns/2/{run_id}/artifacts/{model_name}_model_specification.json"
        # Get the hyperparameter_settings from the model_parameters dictionary
        model_hyperparameter_settings = read_test_model_parameters(file_path, is_baseline_model=True)

        # Get the length of the training window and set the training data accordingly
        rolling_window_length = model_hyperparameter_settings['rolling_window_length']
        # Determine train data set and return it if bayesian model
        train_set = training_data[
            training_data['month_id'].between(last_training_month - rolling_window_length, last_training_month)]

        if not use_fixed_quantiles:
            # Get the quantiles to be computed as estimates
            # Read the 'quantiles' from model_hyperparameter_settings and convert it back to a list
            # Default to an empty list if the key is not found
            quantiles_json = model_hyperparameter_settings.get('quantiles', '[]')
            quantiles = json.loads(quantiles_json)
            # Compute the baseline CRPS score for a country and validation month
        else:
            # Use fixed quantiles from 0.01 to 0.99 in 0.01 steps
            quantiles = [i / 100 for i in range(1, 100)]
        crps_score, quantiles_calculated = compute_baseline_crps(train_set, actual_value, quantiles)
        # Append the CRPS score to the column 'crps_score' in actuals_run_specification
        actuals_run_specification.loc[actuals_run_specification['month_id'] == val_month, 'test_score'] = crps_score
        # Append for one row in competition_contribution_results_country_specific the country_id for column country_id
        # and the val_month for column month_id
        # Addtionally count the no of quantiles and add this value in the column draws and the corresponding quantiles
        # with a own column for each quantile as sample 1, 2, ..., no. draws
        row_temp = {
            "month_id": val_month,
            "country_id": country_id,
            "draw": len(quantiles)
        }

        # Add quantile values as separate columns
        for i, q_value in enumerate(quantiles_calculated['quantile_values']):
            col_name = f"sample_{i + 1}"
            row_temp[col_name] = q_value

        competition_contribution_results_country_specific = competition_contribution_results_country_specific.append(
            row_temp, ignore_index=True)
    # ToDO: Consider adding descriptions what the standard deviations relate to
    crps_results_country_specific = actuals_run_specification[['country_id', 'month_id', 'forecast_horizon',
                                                               'test_score', 'run_id']]

    return crps_results_country_specific, competition_contribution_results_country_specific


def prepare_actual_run_baseline(actuals: pd.DataFrame, best_runs_cross_validation: pd.DataFrame):
    """
    Builds the actuals_run_specification dataframe which contains best model specification for each forecast horizon and
    country with the corresponding actuals data.
    Args:
        actuals: test set after cross-validation process
        best_runs_cross_validation: dataframe containing the best runs of our cross-validation process for each forecast
                                    horizon and country.

    Returns:

    """
    # ToDo: Check if we can leave out country_id
    # Sort best runs dataframe by forecast horizon and country
    best_runs_cross_validation = best_runs_cross_validation.sort_values(
        by=['forecast_horizon', 'country_id']).reset_index(
        drop=True)

    # Sort actuals dataframe by month_id and country_id for sanity
    actuals = actuals.sort_values(by=['month_id', 'country_id']).reset_index(drop=True)
    # ToDo: Check functionality
    # Add a forecast_horizon column to actuals dataframe ranging from 3 to 14
    actuals['forecast_horizon'] = actuals['month_id'] - actuals['month_id'].min() + 3

    # Join best_runs_cross_validation and actuals on forecast_horizon and country_id
    actuals_run_specification = pd.merge(best_runs_cross_validation, actuals, on=['forecast_horizon', 'country_id'])

    return actuals_run_specification


def read_test_model_parameters(file_path: str, is_bayesian_model: Optional[bool] = None,
                               is_baseline_model: Optional[bool] = None) -> (dict, dict):
    """
    Reads the best model parameters obtained from the cross-validation process for either a specific forecast horizon
    (Bayesian model) or a specific forecast horizon and country (baseline model)
    Args:
        file_path: path leading to the JSON file containing the best model parameters
        is_bayesian_model: indicates whether we are reading the best model parameters for the Bayesian model
        is_baseline_model: indicates whether we are reading the best model parameters for the baseline model

    Returns:

    """
    # Read the JSON file
    # toDo: Fix Export in log function
    try:
        with open(file_path, 'r') as f:
            model_parameters = json.load(f)
    except Exception as e:
        print(
            f'An error has occured while reading the JSON file with Excption: {e}. The corresponding values are: \n'
            f'file_path={file_path}')

    if is_bayesian_model:
        stan_data = model_parameters['best_params']
        sampling_parameters = model_parameters['sampling_parameters']
        data_generation_specification = model_parameters['data_generation_specification']
        return stan_data, sampling_parameters, data_generation_specification
    if is_baseline_model:
        model_hyperparameter_settings = model_parameters['best_params']
        return model_hyperparameter_settings


def calculate_crps(idata: az.InferenceData, actuals: pd.DataFrame, posterior_predictive: str) -> (pd.DataFrame, float):
    """
    Calculate the Continuous Ranked Probability Score (CRPS) of a Bayesian model for each observation and the average
    CRPS.

    Parameters:
        idata (az.InferenceData): InferenceData object containing the posterior predictive.
        actuals (pd.DataFrame): DataFrame containing the observed data. Each row corresponds to an observation.
        posterior_predictive: the name of the posterior predictive samples

    Returns:
        crps_scores (pd.DataFrame): A Series containing the CRPS score for each observation.
        average_crps (float): The average CRPS score across all observations.
    """
    try:
        # Access the posterior predictive samples for each observation
        posterior_samples = idata['posterior_predictive'][posterior_predictive].values
    except KeyError:
        raise KeyError(
            f"The InferenceData object does not contain posterior predictive samples {posterior_predictive}.")

    # Reshape the array to a 2D array with samples in rows and observations in columns
    num_chains, num_draws, num_observations = posterior_samples.shape
    reshaped_samples = posterior_samples.reshape(num_chains * num_draws, num_observations)

    # Convert the 2D array to a pandas DataFrame
    df_posterior_samples = pd.DataFrame(reshaped_samples)

    # Optionally, you can add meaningful column names to the DataFrame
    # For example, assuming the original DataFrame has column names 'obs_0', 'obs_1', etc.
    column_names = ['obs_{}'.format(column_idx) for column_idx in range(num_observations)]
    df_posterior_samples.columns = column_names

    # List to store the CRPS scores for each observation
    crps_scores = []

    # Iterate over each column (observation) in the DataFrame
    for col in df_posterior_samples.columns:
        # Get the observation index from the column name
        obs_idx = int(col.split("_")[1])
        observed_values = actuals.ged_sb.iloc[obs_idx]  # Assuming actuals DataFrame has the observed values
        predictive_samples = df_posterior_samples[col]
        predictive_samples = np.array(predictive_samples)

        # Compute CRPS for the current observation
        crps, _, _ = pscore(predictive_samples, observed_values).compute()
        crps_scores.append(crps)

    # Average CRPS across all observations
    average_crps = np.mean(crps_scores)

    return pd.DataFrame(crps_scores), average_crps


def generate_penalty_matrix(no_coefficients):
    """
    Generate the penalty matrix for the Bayesian model which integrated first order random walk regularization in
    spline coefficients.
    Args:
        no_coefficients:

    Returns: penalty_matrix (np.array): penalty matrix for the Bayesian model with shape
                                        (no_coefficients, no_coefficients)

    """
    penalty_matrix = 2 * np.eye(no_coefficients) + (-1) * np.eye(no_coefficients, k=1) + (-1) * np.eye(no_coefficients,
                                                                                                       k=-1)
    penalty_matrix[0, 0] = 1
    penalty_matrix[no_coefficients - 1, no_coefficients - 1] = 1
    return penalty_matrix
