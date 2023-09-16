import json
import time
from typing import Tuple, Optional, Union, Dict, List, Any

import CRPS.CRPS as pscore
import arviz as az
import numpy as np
import optuna
import pandas as pd
import pystan
from numpy import ndarray
from pandas import DataFrame

from data_logging import (mlflow_logging_optuna_bayesian,
                          mlflow_logging_optuna_baseline, ml_flow_logging_actual_baseline)


class HyperparameterTuning:
    def __init__(self,
                 actuals: pd.DataFrame,
                 training_data: pd.DataFrame,
                 k_fold_iterations: int,
                 forecast_horizon: int,
                 target_variable: str,
                 covariates: Optional[List[str]] = None,
                 rolling_window_length: Optional[int] = None,
                 stan_model: Optional[pystan.StanModel] = None,
                 data_generation_specification: Optional[Dict] = None,
                 model_hyperparameters_setting: Optional[Dict] = None,
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
        self.covariates = covariates
        self.rolling_window_length = rolling_window_length
        self.stan_model = stan_model
        self.data_generation_specification = data_generation_specification
        self.model_hyperparameters_setting = model_hyperparameters_setting
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
            covariates=self.covariates,
            rolling_window_length=self.rolling_window_length,
            stan_model=self.stan_model,
            data_generation_specification=self.data_generation_specification,
            model_hyperparameters_setting=self.model_hyperparameters_setting,
            cross_validation_settings=self.cross_validation_settings,
            sampling_parameters=self.sampling_parameters,
            validate_on_horizon=self.validate_on_horizon,
            validate_on_horizon_and_month=self.validate_on_horizon_and_month,
            is_bayesian_model=self.is_bayesian_model,
            is_baseline_model=self.is_baseline_model
        )
        return crps_score  # Optuna will try to minimize this value

    def run_optimization(self, n_trials: int = 100):
        study = optuna.create_study()
        study.optimize(self.objective, n_trials=n_trials)
        return study, study.best_params, study.best_value


# Note: Currently implementation for training on forecast horizons only, also consider the assumptions
def run_model(evaluation_year: int, model_name: str, actuals: pd.DataFrame, training_data: pd.DataFrame,
              model_run_config: dict, optuna_trials: int, validate_on_horizon: bool, validate_on_horizon_and_month: bool,
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

    # Retrieve cross-validation settings from config
    cross_validation_settings = model_run_config['cross_validation_settings']
    # Retrieve K-Fold information from config
    k_fold_iterations = cross_validation_settings['k_fold_iterations']

    # Calculate min and max forecast horizon corresponding to actuals data
    forecast_horizon_min, forecast_horizon_max = calculate_min_max_forecast_horizon(actuals, cross_validation_settings)

    # transform bools into text for mlflow logging later on
    if validate_on_horizon:
        validation_approach = "forecast_horizon"
    else:
        validation_approach = "forecast_horizon_and_month"

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
        (stan_model, sampling_parameters, data_generation_specification, model_hyperparameters_setting,
         data_specification, descriptive_values) = unpack_model_configuration(
            model_run_config, is_bayesian_model=is_bayesian_model)

        # Compile the stan model before running the hyperparameter tuning process
        try:
            stan_model = pystan.StanModel(model_code=stan_model)
        except Exception as e:
            print(
                f'An error has occured while compiling Stan Model with Exception: {e}. The corresponding values are: \n'
                f'stan_model={stan_model}')

        # Iterate over forecast horizons to train the model on different forecast horizons
        for forecast_horizon in range(forecast_horizon_min, forecast_horizon_max + 1):
            # Retrieve target variable from config
            target_variable = data_specification['target_variable']
            # Retrieve covariates from config
            # ToDo: Decide on one approach (data_specification bs data_generation_specification), dependent on how
            #  we solve actuals covariate data issue
            covariates = data_generation_specification['covariates']
            # Calculate the length of the rolling window used in each fold
            rolling_window_length = calculate_rolling_window_length(training_data, k_fold_iterations)
            # Run hyperparameter tuning on forecast horizon
            # ToDo: Check if this works as intended
            # OPTIONAL: Consider logging/ storing the results of the hyperparameter tuning process (bayes)
            tuner = HyperparameterTuning(actuals=actuals,
                                         training_data=training_data,
                                         k_fold_iterations=k_fold_iterations,
                                         forecast_horizon=forecast_horizon,
                                         target_variable=target_variable,
                                         covariates=covariates,
                                         rolling_window_length=rolling_window_length,
                                         stan_model=stan_model,
                                         data_generation_specification=data_generation_specification,
                                         cross_validation_settings=cross_validation_settings,
                                         model_hyperparameters_setting=model_hyperparameters_setting,
                                         sampling_parameters=sampling_parameters,
                                         validate_on_horizon=validate_on_horizon,
                                         validate_on_horizon_and_month=validate_on_horizon_and_month,
                                         is_bayesian_model=is_bayesian_model)

            # Run the optimization and get the study object and best parameters
            study_object, best_params, best_value = tuner.run_optimization(n_trials=optuna_trials)

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

        # ToDo: Implement hyperparameter retrieving using the otuna study objects, either by taking best parameters from
        #       study object or from the exported data in mlflow log (if done), same for baseline
        # Run final model on best parameter configurations obtained from cross-validation process
        actuals_results = compute_stan_model_actual(actuals, training_data, stan_model, best_runs_cross_validation,
                                                    cross_validation_settings)
        return actuals_results, best_runs_cross_validation

    elif is_baseline_model is True:
        # Create empty data frame to store best results of cross validation process
        test_results_all_countries = pd.DataFrame(
            columns=['country_id', 'month_id', 'forecast_horizon', 'test_score', 'run_id'])

        # Unpack the rest of model_run_config dictionary
        (model_hyperparameters_setting, data_specification, descriptive_values) = unpack_model_configuration(
            model_run_config, is_baseline_model=is_baseline_model)

        # Extract country_ids of countries which do have data within the evaluation period
        country_ids = sorted(actuals['country_id'].unique())

        # ToDo: Remove
        # Filter country_ids on countries 133, 124, 220 for testing purposes
        # country_ids = [133, 124, 220]
        # Take last 20 entries of list country_ids
        country_ids = country_ids[-2:]


        # Retrieve target variable from config
        target_variable = data_specification['target_variable']

        # Note: originally loop over horizons and then over countries, which is more performant (less joins, less appends),
        #       however this it's a trade-off between performance and readability
        # Iterate over each country
        for country_id in country_ids:
            # Create empty data frame to store best results of cross validation process for a single country
            best_runs_cross_validation_country = pd.DataFrame(
                columns=["forecast_horizon", "country_id", "run_id", "cross_validation_score"])
            # Select data for country
            actuals_country_specific = actuals[actuals['country_id'] == country_id]
            training_data_country_specific = training_data[training_data['country_id'] == country_id]

            # Iterate over forecast horizons to train the model on different forecast horizons
            for forecast_horizon in range(forecast_horizon_min, forecast_horizon_max + 1):
                # toDo: Check if we need this for the bayesian model as well
                # determine how many training months are needed for the biggest values for rolling window length and
                # forecast horizon


                maximum_months_needed = (model_hyperparameters_setting["rolling_window_length"]["range"][1] +
                                         forecast_horizon +
                                         ((k_fold_iterations) * len(actuals_country_specific)))
                # change k_fold_iterations if data is not sufficient
                if len(training_data_country_specific) < maximum_months_needed:
                    # Determine how many k fold iterations are possible with the available data
                    months_available_for_cv_sets = (len(training_data_country_specific) -
                                                    model_hyperparameters_setting["rolling_window_length"]["range"][1] -
                                                    forecast_horizon -
                                                    # ToDo: Adjustment to reflect missing Nov, Dec for last validation
                                                    #       set. Remove if other approach taken.
                                                    10)
                    k_fold_iterations = months_available_for_cv_sets // len(actuals_country_specific)

                # OPTIONAL: Consider logging/ storing the results of the hyperparameter tuning process (baseline)
                # Run hyperparameter tuning on forecast horizon for each country
                tuner = HyperparameterTuning(actuals=actuals_country_specific,
                                             training_data=training_data_country_specific,
                                             k_fold_iterations=k_fold_iterations,
                                             forecast_horizon=forecast_horizon,
                                             target_variable=target_variable,
                                             model_hyperparameters_setting=model_hyperparameters_setting,
                                             validate_on_horizon=validate_on_horizon,
                                             validate_on_horizon_and_month=validate_on_horizon_and_month,
                                             is_baseline_model=is_baseline_model)

                print(
                    f"Start Hyperparameter tuning for country {country_id} and forecast horizon {forecast_horizon} "
                    f"and evaluation_year {evaluation_year}...")
                # Run the optimization and get the study object and best parameters
                study_object, best_params, best_value = tuner.run_optimization(n_trials=optuna_trials)
                print(
                    f"Hyperparameter tuning for country {country_id} and forecast horizon {forecast_horizon} "
                    f"and evaluation_year {evaluation_year} finished.")

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

            # Run final model on best parameter configurations obtained from cross-validation process
            test_result_country = compute_baseline_model_actual(
                actuals_country_specific, training_data_country_specific, best_runs_cross_validation_country, model_name)
            # Append the results for the country to the results for all countries
            test_results_all_countries = test_results_all_countries.append(test_result_country, ignore_index=True)

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
        run_time_one_evaluation_year = time.time() - start_time
        ml_flow_logging_actual_baseline(model_name=model_name,
                                        evaluation_year=evaluation_year,
                                        process="model-testing",
                                        validation_approach=validation_approach,
                                        test_results_global_dict=test_results_global_dict,
                                        test_results_all_countries=test_results_all_countries,
                                        run_time=run_time_one_evaluation_year)
        # ToDo: Partly redundant return
        return test_results_global_dict, test_results_global, test_results_all_countries


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


def calculate_rolling_window_length(training_data: pd.DataFrame, k: int) -> int:
    """
    Calculate the length of the rolling window for the specified number of folds. The length of the rolling window is
    dependent on the number of folds and corresponds to the maximum number of months that can be used for validating
    first cross-validation set. This is only needed for the Bayesian model.
    Args:
        training_data: DataFrame containing the training data for the cross-validation process
        k: indicates which fold iteration we are in
    Returns:

    """
    # Retrieve last month in training data
    last_observation_month = get_last_observation_month(training_data)

    # Shift the last observation date by K years to determine last training month of first k fold
    last_training_month = last_observation_month - 12 * k

    # Find the first observation date in the training data
    first_observation_month = training_data['month_id'].min()

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
    model_hyperparameters_setting = model_run_config.get("model_hyperparameter_settings", None)
    sampling_parameters = model_run_config.get("sampling_parameters", None)
    data_specification = model_run_config["data_specification"]
    descriptive_values = model_run_config["descriptive_values"]

    if is_bayesian_model is True:
        return (stan_model, sampling_parameters, data_generation_specification, model_hyperparameters_setting,
                data_specification, descriptive_values)
    elif is_baseline_model is True:
        return (model_hyperparameters_setting, data_specification, descriptive_values)


def run_hyperparameter_tuning(trial, actuals: pd.DataFrame, training_data: pd.DataFrame, k_fold_iterations: int,
                              forecast_horizon: int, target_variable: str, covariates: list,
                              rolling_window_length: Optional[int] = None,
                              stan_model: Optional[pystan.StanModel] = None,
                              data_generation_specification: Optional[dict] = None,
                              model_hyperparameters_setting: Optional[dict] = None,
                              cross_validation_settings: Optional[dict] = None,
                              sampling_parameters: Optional[dict] = None, validate_on_horizon: Optional[bool] = None,
                              validate_on_horizon_and_month: Optional[bool] = None,
                              is_bayesian_model: Optional[bool] = None, is_baseline_model: Optional[bool] = None,
                              ) -> float:
    """
    Args:
        cross_validation_settings:
        data_generation_specification:
        covariates:
        trial: optuna trial object
        actuals: test set after cross-validation process
        training_data: DataFrame containing the training data for the cross-validation process
                       (filtered on a country for baseline model)
        k_fold_iterations: indicates the number of folds to be used in the cross-validation process
        forecast_horizon: forecast horizon in months for which we tune the hyperparameters
        target_variable:
        rolling_window_length: Optional parameter as only needed for Bayesian model.
        stan_model:  Optional parameter as only needed for Bayesian model.
        model_hyperparameters_setting: dict containing information about the hyperparameters to be tuned
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
    for k in range(1, k_fold_iterations+1):
        # Retrieve the validation set for the current fold
        validation_set = get_validation_set_for_fold(actuals, training_data, k, forecast_horizon, target_variable,
                                                     covariates, validate_on_horizon, validate_on_horizon_and_month,
                                                     is_baseline_model=is_baseline_model,
                                                     is_bayesian_model=is_bayesian_model)

        # Generate empty list to store the CRPS scores for the validation set
        crps_scores_val_set = []
        # Iterate over the months to be forecasted
        for val_month in validation_set['month_id']:
            # Train model on training set and cross-validate on validation month using the specified model
            if is_bayesian_model is True:
                # Retrieve the training set corresponding to current validation month
                train_set = get_train_set_for_fold(training_data, val_month, forecast_horizon, rolling_window_length,
                                                   target_variable, covariates, is_bayesian_model=is_bayesian_model)
                crps_val_month = compute_stan_model(trial, stan_model, data_generation_specification,
                                                    cross_validation_settings, model_hyperparameters_setting,
                                                    sampling_parameters, train_set, val_month, target_variable,
                                                    covariates)
            if is_baseline_model is True:
                # Note: as for the baseline the length of the train set is a hyperparameter itself, it its specified
                #       within the compute_baseline_model function
                crps_val_month = compute_baseline_model(trial, training_data, val_month, model_hyperparameters_setting,
                                                        forecast_horizon, target_variable)
            # Append the CRPS score to the list
            crps_scores_val_set.append(crps_val_month)
        # Depending on the validation approach crps_scores_val_set contains the CRPS scores for a single month or
        # similar to number of months in actual
        # Calculate the average CRPS score for the validation set
        average_crps_val_set = np.mean(crps_scores_val_set)
        # Append the average CRPS score of one fold to the k_fold_iterations list
        crps_scores_k_fold_iterations.append(average_crps_val_set)

    # Calculate the average and standard deviation of the CRPS scores across all folds
    average_crps_k_fold_iterations = np.mean(crps_scores_k_fold_iterations)

    return average_crps_k_fold_iterations


# ToDo: Inspect if it's sufficient to return the months as set because we obtain the actual value from training data in
#       cross validation process
def get_validation_set_for_fold(actuals: pd.DataFrame, training_data: pd.DataFrame, k: int, forecast_horizon: int,
                                target_variable, covariates: list, validate_on_horizon: Optional[bool] = None,
                                validate_on_horizon_and_month: Optional[bool] = None,
                                is_baseline_model: Optional[bool] = None,
                                is_bayesian_model: Optional[bool] = None) -> pd.DataFrame:
    """
    Get the validation set for the specified fold.
    Note: If we validate on forecast horizon and month this function will return a single month only.
    Args:
        covariates:
        target_variable:
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
    first_validation_month = actuals['month_id'].min() - 12 * k - 12

    if validate_on_horizon is True:
        # Last month in validation set
        last_validation_month = actuals['month_id'].max() - 12 * k - 12
        # Get validation set
        validation_set = training_data[training_data['month_id'].between(first_validation_month, last_validation_month)]

    # ToDo: Inspect if works as intended
    if validate_on_horizon_and_month is True:
        # Calculate the validation month in accordance to forecast horizon
        validation_month = first_validation_month + forecast_horizon - 3
        # Get validation set
        validation_set = training_data[training_data['month_id'] == validation_month]

    # ToDo: Check if its suitable to drop the month_id column
    if is_bayesian_model is True:
        return validation_set[['month_id'] + [target_variable] + covariates]
    elif is_baseline_model is True:
        return validation_set[['month_id'] + [target_variable]]


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
                       cross_validation_settings: dict, model_hyperparameters_setting: dict, sampling_parameters: dict,
                       train_set: pd.DataFrame, validation_month: int, target_variable: str, covariates: list) -> float:
    """
    Compute the Bayesian model with the hyperparamters delivered by Optuna.
    Args:
        cross_validation_settings:
        target_variable:
        covariates:
        data_generation_specification:
        trial:
        stan_model:
        model_hyperparameters_setting:
        sampling_parameters:
        train_set:
        validation_month: Single value of validation month

    Returns:

    """
    stan_data = {}
    # Get the target data and derive the needed stats such as length of the data
    dataY = train_set[target_variable].values
    dataY_eval = validation_month[target_variable].values
    stan_data["Y"] = dataY
    stan_data["Y_eval"] = dataY_eval
    stan_data["no_data"] = len(dataY)
    stan_data["no_data_eval"] = len(dataY_eval)
    # Get the static parameters of the configuration dictionary and add them to the stan_data dictionary
    stan_data["spline_degree"] = data_generation_specification['spline_degree']

    # Get the dynamic parameters of the configuration dictionary, initialize them as hyperparameters and add them to
    # the stan_data dictionary
    for key, value in model_hyperparameters_setting.items():
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

    # Specify covariate data passed to stan model and parameters subject to other hyperparameters
    covariate_iterator = 1
    for covariate in covariates:
        stan_data[f"covariate_data_X{covariate_iterator}"] = train_set[covariate].values
        stan_data[f"covariate_data_X{covariate_iterator}_eval"] = validation_month[covariate].values
        stan_data[f"num_basis_X{covariate_iterator}"] = (
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


def compute_baseline_model(trial, training_data: pd.DataFrame, validation_month: int,
                           model_hyperparameters_setting: dict, forecast_horizon: int, target_variable: str,
                           is_baseline_model=True) -> float:
    """
    Compute the baseline model for the specified configuration.
    Args:
        is_baseline_model:
        target_variable:
        trial:
        forecast_horizon:
        training_data: DataFrame containing the training data for the cross-validation process
        model_hyperparameters_setting: dictionary with hyperparameter settings for baseline model
        validation_month: month_id of validation month

    Returns:

    """
    # Get the hyperparameters of the configuration dictionary
    rolling_window_length = trial.suggest_int('rolling_window_length',
                                              model_hyperparameters_setting['rolling_window_length']['range'][0],
                                              model_hyperparameters_setting['rolling_window_length']['range'][1])
    n_quantiles = trial.suggest_int('n_quantiles',
                                    model_hyperparameters_setting['quantiles']['n_quantiles']['range'][0],
                                    model_hyperparameters_setting['quantiles']['n_quantiles']['range'][1])
    quantiles = []
    for i in range(n_quantiles):
        quantile = trial.suggest_float(f'quantile_{i + 1}',
                                       model_hyperparameters_setting['quantiles']['quantile_value']['range'][0],
                                       model_hyperparameters_setting['quantiles']['quantile_value']['range'][1],
                                       step=model_hyperparameters_setting['quantiles']['quantile_value']['step'])
        quantiles.append(quantile)
    # Sort the quantiles to ensure they are in ascending order
    quantiles = sorted(quantiles)
    # Compute the training set for the current validation month
    train_set = get_train_set_for_fold(training_data=training_data,
                                       validation_month=validation_month,
                                       forecast_horizon=forecast_horizon,
                                       rolling_window_length=rolling_window_length,
                                       target_variable=target_variable,
                                       is_baseline_model=is_baseline_model)
    observation = training_data[training_data['month_id'] == validation_month][target_variable].values[0]
    crps = compute_baseline_crps(train_set, observation, quantiles)

    return crps


def compute_baseline_crps(train_set: pd.DataFrame, observation: float, quantiles: list) -> float:
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
            stan_data[f"num_basis_X{covariate_iterator}"] = (
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
                                  best_runs_cross_validation: pd.DataFrame, model_name: str) -> (float, float, pd.DataFrame):
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

    actuals_run_specification = prepare_actual_run_baseline(actuals, best_runs_cross_validation)
    # Run baseline model with the best parameter settings for each forecast horizon
    for index, row in actuals_run_specification.iterrows():
        # Note: we don't need tp access the forecast horizon explicitly as it is given by the validation month implicitly
        run_id = row['run_id']
        val_month = row['month_id']
        actual_value = row['ged_sb']

        # Construct the file path the parameter settings are stored in
        # ToDo: Check that baseline projects are stored under 2
        file_path = f"mlruns/2/{run_id}/artifacts/{model_name}_model_specification.json"
        # Get the hyperparameters_setting from the model_parameters dictionary
        model_hyperparameters_setting = read_test_model_parameters(file_path, is_baseline_model=True)

        # Get the length of the training window and set the training data accordingly
        rolling_window_length = model_hyperparameters_setting['rolling_window_length']
        # Determine last month of training data
        last_training_month = (
                actuals_run_specification['month_id'].min() - actuals_run_specification['forecast_horizon'].min())
        # Determine train data set and return it if bayesian model
        train_set = training_data[
            training_data['month_id'].between(last_training_month - rolling_window_length, last_training_month)]

        # Get the quantiles to be computed as estimates
        # Read the 'quantiles' from model_hyperparameters_setting and convert it back to a list
        # Default to an empty list if the key is not found
        quantiles_json = model_hyperparameters_setting.get('quantiles', '[]')
        quantiles = json.loads(quantiles_json)
        # Compute the baseline CRPS score for a country and validation month
        crps_score = compute_baseline_crps(train_set, actual_value, quantiles)
        # Append the CRPS score to the column 'crps_score' in actuals_run_specification
        actuals_run_specification.loc[actuals_run_specification['month_id'] == val_month, 'test_score'] = crps_score

    # ToDO: Consider adding descriptions what the standard deviations relate to
    crps_results_country_specific = actuals_run_specification[['country_id', 'month_id', 'forecast_horizon',
    'test_score', 'run_id']]

    return crps_results_country_specific


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
        model_hyperparameters_setting = model_parameters['best_params']
        return model_hyperparameters_setting


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
