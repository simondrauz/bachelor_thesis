import json
import os
import tempfile
from typing import List, Tuple
from typing import Optional
from typing import Union

import arviz as az
import mlflow
import optuna
import pandas as pd
from help_functions import map_country_id_to_country_name

from help_functions import seconds_to_format


def count_divergences(idata: az.InferenceData) -> int:
    """
    Count the number of divergent transitions in an InferenceData object.

    Args:
        idata (az.InferenceData): ArviZ InferenceData object.

    Returns:
        int: Number of divergent transitions.
    """
    # Access the divergent transitions information
    diverging = idata.sample_stats['diverging'].values

    # Sum up the True values to get the count of divergences
    return diverging.sum()


# Optional: Include data_characteristics of evaluation data (in pipeline it would be the characteristics of the
#  concatenation of k cross-validation sets)
def generate_data_dict(training_data: pd.DataFrame,
                       evaluation_data: pd.DataFrame,
                       data_characteristics_train: dict,
                       data_characteristics_eval: dict,
                       data_transformation: str,
                       country_name: str, predictors: List,
                       is_bayesian_model: Optional[bool] = None,
                       is_baseline_model: Optional[bool] = None) -> dict:
    """
    Information about data used in model.
    Args:
        training_data:
        evaluation_data:
        data_characteristics_train: dictionary containing information about training data
        data_characteristics_eval:
        data_transformation: information about data transformation
        country_name:
        predictors:

    Returns:

    """
    if is_bayesian_model:
        return {
            "training_data": training_data,
            "evaluation_data": evaluation_data,
            "data_characteristics_train": data_characteristics_train,
            "data_characteristics_eval": data_characteristics_eval,
            "data_transformation": data_transformation,
            "predictors": predictors
        }
    if is_baseline_model:
        return {
            "training_data": training_data,
            "evaluation_data": evaluation_data,
            "data_characteristics_train": data_characteristics_train,
            "data_characteristics_eval": data_characteristics_eval,
            "data_transformation": data_transformation,
            "country_name": country_name,
        }


def generate_model_dict(model_name: str,
                        prior_specifications: dict,
                        is_bayesian_model: Optional[bool] = None,
                        is_baseline_model: Optional[bool] = None
                        ) -> dict:
    """
    Information about model type and specifications.
    Args:
        model_name: name suitable for a file to store in
        prior_specifications: dictionary containing assumptions for prior parameter distributions

    Returns:

    """
    if is_bayesian_model:
        return {
            "model_name": model_name,
            "prior_specifications": prior_specifications
        }
    if is_baseline_model:
        return {
            "model_name": model_name,
        }


# ToDo: Extend this dictionary to store hyperparameters of baseline model
def generate_model_hyperparameters_dict(predictors: List,
                                        intercept_hyperparameters: Tuple[float, float],
                                        alpha_nb_hyperparameters: Union[None, float, Tuple[float, float]] = None,
                                        no_spline_coefficients_per_regressor: Optional[List[int]] = None,
                                        tau_hyperparameters: Optional[List[Tuple[float, float]]] = None,
                                        beta_hyperparameters: Optional[List[Tuple[float, float]]] = None,
                                        sigma_gaussian_hyperparameters: Optional[List[Tuple[float, float]]] = None
                                        ) -> dict:
    """
    Information about model hyperparameters.
    Args:
        predictors: List of predictors used in model for clear assignment of hyperparameters to predictors
        no_spline_coefficients_per_regressor: no. of knots corresponding to each predictor in List 'predictors', as List with each
                                element being an integer corresponding to a predictor
        tau_hyperparameters: alpha and beta hyperparameters for scaling factor tau corresponding to each predictor in
                             List 'predictors', as List with each element being a Tuple of alpha and beta
                             hyperparameters corresponding to a predictor
        alpha_nb_hyperparameters: alpha and beta hyperparameters for Gamma distribution of alpha parameter of
                                  negative binomial distribution
        intercept_hyperparameters: mu and sigma for normal distribution of intercept
        beta_hyperparameters: parameters for normal distributions of regression coefficients
        sigma_gaussian_hyperparameters: parameters for normal distributions of sigma if likelihood is gaussian

    Returns:

    """
    # First, start with the base dictionary that always gets returned.
    result_dict = {
        "alpha_nb_hyperparameters": alpha_nb_hyperparameters,
        "intercept_hyperparameters": intercept_hyperparameters
    }

    if alpha_nb_hyperparameters is not None:
        result_dict["alpha_nb_hyperparameters"] = alpha_nb_hyperparameters
    # Check if no_spline_coefficients_per_regressor is not None, assert its length, and then add it to the dictionary.
    if no_spline_coefficients_per_regressor is not None:
        assert len(predictors) == len(
            no_spline_coefficients_per_regressor), \
            "Mismatch in length between predictors and no_spline_coefficients_per_regressor"
        result_dict["no_spline_coefficients_per_regressor"] = {
            predictors[i]: no_spline_coefficients_per_regressor[i] for i in range(len(predictors))}

    # Similarly, check if tau_hyperparameters is not None, assert its length, and then add it to the dictionary.
    if tau_hyperparameters is not None:
        assert len(predictors) == len(
            tau_hyperparameters), "Mismatch in length between predictors and tau_hyperparameters"
        result_dict["tau_hyperparameters"] = {predictors[i]: tau_hyperparameters[i] for i in range(len(predictors))}

    # Similarly, check if beta_hyperparameters is not None, assert its length, and then add it to the dictionary.
    if beta_hyperparameters is not None:
        result_dict["beta_hyperparameters"] = beta_hyperparameters

    # Similarly, check if sigma_gaussian_hyperparameters is not None, assert its length, and then add it to
    # the dictionary.
    if sigma_gaussian_hyperparameters is not None:
        result_dict["sigma_gaussian_hyperparameters"] = sigma_gaussian_hyperparameters

    return result_dict


def generate_sampling_hyperparameters_dict(tuning_iterations: int,
                                           sampling_iterations: int,
                                           target_acceptance_rate: float,
                                           chains: int
                                           ) -> dict:
    """
    Information about sampling hyperparameters.
    Args:
        tuning_iterations: no. of tuning iterations
        sampling_iterations: no. of sampling iterations
        target_acceptance_rate:
        chains: no. of chains used in samoing process

    Returns:

    """
    return {
        "tuning_iterations": tuning_iterations,
        "sampling_iterations": sampling_iterations,
        "target_acceptance_rate": target_acceptance_rate,
        "chains": chains
    }


def generate_sampling_monitoring_dict(idata: az.InferenceData, sampling_time: float) -> dict:
    """
    Information about monitoring characteristics of sampling process.
    Args:
        idata: Inference Data object for counting no. of divergences in sampling process
        sampling_time: time taken to sample pymc model (measured with time.time)

    Returns:

    """
    divergences = count_divergences(idata)
    return {
        "divergences": divergences,
        "sampling_time": seconds_to_format(sampling_time)
    }


def generate_model_results_dict(idata: az.InferenceData,
                                crps_score_train: float,
                                crps_score_eval: float,
                                is_bayesian_model: Optional[bool] = None,
                                is_baseline_model: Optional[bool] = None) -> dict:
    """
    Information about model results.
    Args:
        idata: InferenceData object containing posterior samples, sample iterations, posterior probabilities
        crps_score_train: average crps score of model over validation set
        crps_score_eval: average crps score of model over evaluation set

    Returns:

    """
    if is_bayesian_model:
        return {
            "idata": idata,
            "crps_score_train": crps_score_train,
            "crps_score_eval": crps_score_eval
        }
    if is_baseline_model:
        return {
            "crps_score_eval": crps_score_eval
        }


def generate_all_dictionaries(training_data: pd.DataFrame, evaluation_data: pd.DataFrame,
                              data_characteristics_train: dict, data_characteristics_eval: dict,
                              data_transformation: str,
                              optuna_logging: bool,
                              optuna_best_params: Optional[dict] = None,
                              predictors: Optional[List] = None,
                              model_name: Optional[str] = None,
                              prior_specifications: Optional[dict] = None,
                              tuning_iterations: Optional[int] = None,
                              sampling_iterations: Optional[int] = None,
                              target_acceptance_rate: Optional[float] = None,
                              chains: Optional[int] = None,
                              sampling_time: Optional[float] = None,
                              idata: Optional[az.InferenceData] = None,
                              crps_score_train: Optional[float] = None,
                              crps_score_eval: Optional[float] = None,
                              country_name: Optional[str] = None,
                              intercept_hyperparameters: Optional[Tuple[float, float]] = None,
                              alpha_nb_hyperparameters: Optional[Tuple[float, float]] = None,
                              no_spline_coefficients_per_regressor: Optional[List[int]] = None,
                              tau_hyperparameters: Optional[List[Tuple[float, float]]] = None,
                              beta_hyperparameters: Optional[Tuple[float, float]] = None,
                              sigma_gaussian_hyperparameters: Optional[float] = None,
                              is_bayesian_model: Optional[bool] = None,
                              is_baseline_model: Optional[bool] = None
                              ) -> dict:
    """
    Generate all dictionaries for pymc model.
    Args:
        training_data:
        evaluation_data:
        data_characteristics_train: of training data
        data_characteristics_eval:
        data_transformation:
        country_name:
        predictors:
        model_name:
        prior_specifications:
        alpha_nb_hyperparameters:
        intercept_hyperparameters:
        tuning_iterations:
        sampling_iterations:
        target_acceptance_rate:
        chains:
        sampling_time:
        idata:
        crps_score_train:
        crps_score_eval:
        no_spline_coefficients_per_regressor:
        tau_hyperparameters:
        beta_hyperparameters:
        sigma_gaussian_hyperparameters:

    Returns:

    """
    data_dict = generate_data_dict(training_data=training_data,
                                   evaluation_data=evaluation_data,
                                   data_characteristics_train=data_characteristics_train,
                                   data_characteristics_eval=data_characteristics_eval,
                                   data_transformation=data_transformation,
                                   country_name=country_name,
                                   predictors=predictors,
                                   is_bayesian_model=is_bayesian_model, is_baseline_model=is_baseline_model)
    model_dict = generate_model_dict(model_name=model_name,
                                     prior_specifications=prior_specifications,
                                     is_bayesian_model=is_bayesian_model, is_baseline_model=is_baseline_model)
    if optuna_logging is True:
        model_hyperparameters_dict = optuna_best_params
    else:
        model_hyperparameters_dict = generate_model_hyperparameters_dict(
            predictors=predictors,
            intercept_hyperparameters=intercept_hyperparameters,
            alpha_nb_hyperparameters=alpha_nb_hyperparameters,
            no_spline_coefficients_per_regressor=no_spline_coefficients_per_regressor,
            tau_hyperparameters=tau_hyperparameters,
            beta_hyperparameters=beta_hyperparameters,
            sigma_gaussian_hyperparameters=sigma_gaussian_hyperparameters)
    if is_bayesian_model:
        sampling_hyperparameters_dict = generate_sampling_hyperparameters_dict(
            tuning_iterations=tuning_iterations,
            sampling_iterations=sampling_iterations,
            target_acceptance_rate=target_acceptance_rate,
            chains=chains)
        sampling_monitoring_dict = generate_sampling_monitoring_dict(idata=idata,
                                                                     sampling_time=sampling_time)
    model_results_dict = generate_model_results_dict(idata=idata,
                                                     crps_score_train=crps_score_train,
                                                     crps_score_eval=crps_score_eval,
                                                     is_bayesian_model=is_bayesian_model,
                                                     is_baseline_model=is_baseline_model)

    if is_bayesian_model:
        return {
            "data": data_dict,
            "model": model_dict,
            "hyperparameters_model": model_hyperparameters_dict,
            "hyperparameters_sampling": sampling_hyperparameters_dict,
            "sampling_monitoring": sampling_monitoring_dict,
            "model_results": model_results_dict
        }
    if is_baseline_model:
        return {
            "data": data_dict,
            "model": model_dict,
            "hyperparameters_model": model_hyperparameters_dict,
            "model_results": model_results_dict
        }


# ToDo: Logging of CRPS evaluation score
def ml_flow_tracking(data: dict,
                     model: dict,
                     hyperparameters_model: dict,
                     model_results: dict,
                     hyperparameters_sampling: Optional[dict] = None,
                     sampling_monitoring: Optional[dict] = None,
                     is_bayesian_model: Optional[bool] = None,
                     is_baseline_model: Optional[bool] = None):
    """
    Preliminary function to track hyperparameters and results of pymc model
    Args:
        data: training data, evaluation_data, data characteristics, data transformation, country name, predictors
        model: model description, prior specifications
        hyperparameters_model: no. of spline regression coefficients per regressor, tau hyperparameters, alpha hyperparameters,
                               intercept hyperparameters
        hyperparameters_sampling: tuning iterations, sampling iterations, target acceptance rate, chains
        sampling_monitoring: divergences, sampling time
        model_results: InferenceData, crps score

    Returns:

    """
    with mlflow.start_run(run_name=model["model_name"]):

        # Create a temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            # Log data parameters
            for key, value in data.items():
                if key == 'training_data':
                    try:
                        # Log only training_data as pandas dataframe
                        training_data_path = os.path.join(tmpdir, f"{model['model_name']}_training_data.parquet")
                        data["training_data"].to_parquet(training_data_path)
                    except Exception as e:
                        print(f"Error storing training data as parquet file: {e}")
                elif key == 'evaluation_data':
                    try:
                        # Log only training_data as pandas dataframe
                        evaluation_data_path = os.path.join(tmpdir, f"{model['model_name']}_evaluation_data.parquet")
                        data["evaluation_data"].to_parquet(evaluation_data_path)
                    except Exception as e:
                        print(f"Error storing evaluation data as parquet file: {e}")
                elif key == 'data_characteristics_train' or key == 'data_characteristics_eval':
                    for char_key, char_value in value.items():
                        # Convert the dictionary to a JSON string
                        serialized_char_value = json.dumps(char_value)

                        mlflow.log_param(f'{char_key}_{key}', serialized_char_value)
                else:
                    mlflow.log_param(key, str(value))

            # Log model parameters
            for key, value in model.items():
                mlflow.log_param(key, str(value))

            # Log model hyperparameters
            for key, value in hyperparameters_model.items():
                if isinstance(value, dict):
                    value = json.dumps(value)
                mlflow.log_param(key, str(value))

            if is_bayesian_model:
                # Log sampling hyperparameters
                for key, value in hyperparameters_sampling.items():
                    mlflow.log_param(key, str(value))

                # Log sampling monitoring
                for key, value in sampling_monitoring.items():
                    mlflow.log_param(key, str(value))

            # Log CRPS scores
            mlflow.log_metric('crps_train', model_results["crps_score_train"])
            mlflow.log_metric('crps_eval', model_results["crps_score_eval"])

            # Export and log the InferenceData as an artifact
            try:
                # Logs only InferenceData object, not the whole artifact folder
                idata_path = os.path.join(tmpdir, f"{model['model_name']}_results.nc")
                model_results["idata"].to_netcdf(idata_path)
            except Exception as e:
                print(f"Error storing InferenceData as json file: {e}")
            try:
                # Log files stored in artifact folder of current run
                mlflow.log_artifacts(tmpdir)
            except Exception as e:
                print(f"Error logging artifacts: {e}")


def mlflow_logging_optuna_bayesian(model_name: str,
                                   evaluation_year: int,
                                   study_object: optuna.Study,
                                   sampling_parameters: dict,
                                   forecast_horizon: int,
                                   process: str,
                                   data_generation_specification: dict,
                                   validation_approach: str):
    # Check that process is either 'cross-validation' or 'model-testing'
    assert process in ['cross-validation', 'model-testing'], \
        "process variable must be either 'cross-validation' or 'model-testing'"
    best_params = study_object.best_params
    best_value = study_object.best_value
    with mlflow.start_run(run_name=model_name) as run:
        # Get the run_id of the current run
        run_id = run.info.run_id
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            # Log conceptual parameters
            mlflow.log_param('model_name', model_name)
            mlflow.log_param('evaluation_year', evaluation_year)
            mlflow.log_param('process', process)
            mlflow.log_param('validation_approach', validation_approach)
            mlflow.log_param('forecast_horizon', forecast_horizon)

            # Log data generation specification parameters
            mlflow.log_params(data_generation_specification)
            # Log hyperparameters of best_params structure as obtained from optuna.Study.best_params
            mlflow.log_params(best_params)

            # Log sampling parameters
            mlflow.log_params(sampling_parameters)

            # Log CRPS score corresponding to best_params
            mlflow.log_metric('CRPS', best_value)

            # Create a dictionary with best_params and sampling_parameters
            model_specification = {
                "best_params": best_params,
                "sampling_parameters": sampling_parameters,
                "data_generation_specification": data_generation_specification
            }

            # Log the model_specification dict as a JSON string as an artifact
            try:
                dict_path = os.path.join(tmpdir, f"{model_name}_model_specification.json")
                model_specification.to_json(dict_path)
                mlflow.log_artifact(tmpdir)
            except Exception as e:
                print("An error occured while logging the model_specifications as an artifact: ", e)

    return run_id


def mlflow_logging_optuna_baseline(model_name: str,
                                   evaluation_year: int,
                                   country_id: int,
                                   country_mapping: pd.DataFrame,
                                   k_fold_iterations: int,
                                   study_object: optuna.Study,
                                   forecast_horizon: int,
                                   process: str,
                                   validation_approach: str):
    # Check that process is either 'cross-validation' or 'model-testing'
    assert process in ['cross-validation', 'model-testing'], \
        "process variable must be either 'cross-validation' or 'model-testing'"
    best_params = study_object.best_params
    best_value = study_object.best_value
    country_name = map_country_id_to_country_name(country_id=country_id, country_mapping=country_mapping)
    mlflow.set_experiment("baseline_model")
    with mlflow.start_run(run_name=model_name) as run:
        # Get the run_id of the current run
        run_id = run.info.run_id
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            # Log conceptual parameters
            mlflow.log_param('model_name', model_name)
            mlflow.log_param('evaluation_year', evaluation_year)
            mlflow.log_param('process', process)
            mlflow.log_param('validation_approach', validation_approach)
            mlflow.log_param('forecast_horizon', forecast_horizon)
            mlflow.log_param('country_name', country_name)
            mlflow.log_param('k_fold_iterations', k_fold_iterations)

            # Extract quantile_X entries and convert them to a list
            quantile_keys = [key for key in best_params.keys() if 'quantile_' in key]
            quantile_values = [best_params[key] for key in sorted(quantile_keys)]

            # Sort the list of quantile values
            quantile_values.sort()
            # Convert the sorted list to a JSON-formatted string
            quantile_values_json = json.dumps(quantile_values)

            # Remove the individual quantile_X entries from best_params
            for key in quantile_keys:
                del best_params[key]

            # Add the quantile list back as a single entry
            best_params['quantiles'] = quantile_values_json

            # Log hyperparameters of best_params structure as obtained from optuna.Study.best_params
            mlflow.log_params(best_params)

            # Log CRPS score corresponding to best_params
            mlflow.log_metric('CRPS', best_value)

            # Create a dictionary with best_params and
            model_specification = {
                "best_params": best_params,
            }

            # Log the model_specification dict as a JSON string as an artifact
            try:
                dict_path = os.path.join(tmpdir, f"{model_name}_model_specification.json")

                # Save the dictionary as a JSON file
                with open(dict_path, 'w') as f:
                    json.dump(model_specification, f)

                mlflow.log_artifact(dict_path)  # Log the JSON file as an artifact
            except Exception as e:
                print("An error occured while logging the model_specifications as an artifact: ", e)

    return run_id


def ml_flow_logging_actual_baseline(model_name: str,
                                    evaluation_year: int,
                                    process: str,
                                    validation_approach: str,
                                    test_results_global_dict: dict,
                                    test_results_all_countries: pd.DataFrame,
                                    run_time: float):
    # Check that process is either 'cross-validation' or 'model-testing'
    assert process in ['cross-validation', 'model-testing'], \
        "process variable must be either 'cross-validation' or 'model-testing'"
    average_crps = test_results_global_dict["average_global_test_score"]
    std_crps = test_results_global_dict["std_global_test_score"]
    test_results_global = test_results_global_dict["test_results_global"]
    run_time_formatted = seconds_to_format(run_time)
    mlflow.set_experiment("baseline_model_actual")
    with mlflow.start_run(run_name=model_name) as run:
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            # Log conceptual parameters
            mlflow.log_param('model_name', model_name)
            mlflow.log_param('evaluation_year', evaluation_year)
            mlflow.log_param('process', process)
            mlflow.log_param('validation_approach', validation_approach)
            mlflow.log_param('run_time', run_time_formatted)
            mlflow.log_metric('average_crps', average_crps)
            mlflow.log_metric('std_crps', std_crps)

            try:
                # Save the DataFrame to a Parquet file
                test_results_global_path = os.path.join(tmpdir, f"{model_name}_results_actuals_{evaluation_year}.parquet")
                test_results_global.to_parquet(test_results_global_path)

                # Log the Parquet file as an artifact
                mlflow.log_artifact(test_results_global_path)
            except Exception as e:
                print("An error occurred while logging the actuals results as an artifact: ", e)

            try:
                # Save the DataFrame to a Parquet file
                test_results_all_countries_path = os.path.join(tmpdir, f"{model_name}_results_all_countries{evaluation_year}.parquet")
                test_results_all_countries.to_parquet(test_results_all_countries_path)

                # Log the Parquet file as an artifact
                mlflow.log_artifact(test_results_all_countries_path)
            except Exception as e:
                print("An error occurred while logging the actuals results as an artifact: ", e)
