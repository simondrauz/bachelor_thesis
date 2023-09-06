import json
import os

import CRPS.CRPS as pscore
import arviz as az
import mlflow
import numpy as np
import pandas as pd
from typing import Optional
from typing import Union

from help_functions import seconds_to_format
from typing import List, Tuple
import tempfile


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
    penalty_matrix = 2 * np.eye(no_coefficients) + (-1) * np.eye(no_coefficients, k=1) + (-1) * np.eye(no_coefficients,
                                                                                                       k=-1)
    penalty_matrix[0, 0] = 1
    penalty_matrix[no_coefficients - 1, no_coefficients - 1] = 1
    return penalty_matrix


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
def generate_data_dict(training_data: pd.DataFrame, evaluation_data: pd.DataFrame, data_characteristics_train: dict,
                       data_characteristics_eval, data_transformation: str,

                       country_name: str, predictors: List) -> dict:
    """
    Information about data used in model.
    Args:
        training_data:
        evaluation_data:
        data_characteristics_train: dictionary containing information about training data
        data_transformation: information about data transformation
        country_name:
        predictors:

    Returns:

    """
    return {
        "training_data": training_data,
        "evaluation_data": evaluation_data,
        "data_characteristics_train": data_characteristics_train,
        "data_characteristics_eval": data_characteristics_eval,
        "data_transformation": data_transformation,
        "country_name": country_name,
        "predictors": predictors
    }


def generate_model_dict(model_name: str, prior_specifications: dict) -> dict:
    """
    Information about model type and specifications.
    Args:
        model_name: name suitable for a file to store in
        prior_specifications: dictionary containing assumptions for prior parameter distributions

    Returns:

    """
    return {
        "model_name": model_name,
        "prior_specifications": prior_specifications
    }


def generate_model_hyperparameters_dict(predictors: List,
                                        intercept_hyperparameters: Tuple[float, float],
                                        alpha_nb_hyperparameters: Union[None, float, Tuple[float, float]] = None,
                                        no_spline_coefficients_per_regressor: Optional[List[int]] = None,
                                        tau_hyperparameters: Optional[List[Tuple[float, float]]] = None,
                                        beta_hyperparameters: Optional[List[Tuple[float, float]]] = None,
                                        sigma_gaussian_hyperparameters: Optional[List[Tuple[float, float]]] = None) \
        -> dict:
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


def generate_sampling_hyperparameters_dict(tuning_iterations: int, sampling_iterations: int,
                                           target_acceptance_rate: float, chains: int) -> dict:
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


def generate_sampling_monitoring_dict(divergences: int, sampling_time: float) -> dict:
    """
    Information about monitoring characteristics of sampling process.
    Args:
        divergences: no. of divergences in sampling process
        sampling_time: time taken to sample pymc model (measured with time.time)

    Returns:

    """
    return {
        "divergences": divergences,
        "sampling_time": seconds_to_format(sampling_time)
    }


def generate_model_results_dict(idata: az.InferenceData, crps_score_train: float, crps_score_eval: float) -> dict:
    """
    Information about model results.
    Args:
        idata: InferenceData object containing posterior samples, sample iterations, posterior probabilities
        crps_score_train: average crps score of model over validation set
        crps_score_eval: average crps score of model over evaluation set

    Returns:

    """
    return {
        "idata": idata,
        "crps_score_train": crps_score_train,
        "crps_score_eval": crps_score_eval
    }


def generate_all_dictionaries(training_data: pd.DataFrame, evaluation_data: pd.DataFrame,
                              data_characteristics_train: dict, data_characteristics_eval: dict,
                              data_transformation: str,
                              country_name: str, predictors: List,
                              model_name: str, prior_specifications: dict,
                              intercept_hyperparameters: Tuple[float, float], tuning_iterations: int,
                              sampling_iterations: int, target_acceptance_rate: float, chains: int,
                              divergences: int, sampling_time: float, idata: az.InferenceData,
                              crps_score_train: float, crps_score_eval: float,
                              alpha_nb_hyperparameters: Optional[Tuple[float, float]] = None,
                              no_spline_coefficients_per_regressor: Optional[List[int]] = None,
                              tau_hyperparameters: Optional[List[Tuple[float, float]]] = None,
                              beta_hyperparameters: Optional[Tuple[float, float]] = None,
                              sigma_gaussian_hyperparameters: Optional[float] = None
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
        divergences:
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
    data_dict = generate_data_dict(training_data, evaluation_data, data_characteristics_train,
                                   data_characteristics_eval,
                                   data_transformation, country_name, predictors)
    model_dict = generate_model_dict(model_name, prior_specifications)
    model_hyperparameters_dict = generate_model_hyperparameters_dict(predictors,
                                                                     intercept_hyperparameters,
                                                                     alpha_nb_hyperparameters,
                                                                     no_spline_coefficients_per_regressor,
                                                                     tau_hyperparameters,
                                                                     beta_hyperparameters,
                                                                     sigma_gaussian_hyperparameters)
    sampling_hyperparameters_dict = generate_sampling_hyperparameters_dict(tuning_iterations, sampling_iterations,
                                                                           target_acceptance_rate, chains)
    sampling_monitoring_dict = generate_sampling_monitoring_dict(divergences, sampling_time)
    model_results_dict = generate_model_results_dict(idata, crps_score_train, crps_score_eval)

    return {
        "data": data_dict,
        "model": model_dict,
        "hyperparameters_model": model_hyperparameters_dict,
        "hyperparameters_sampling": sampling_hyperparameters_dict,
        "sampling_monitoring": sampling_monitoring_dict,
        "model_results": model_results_dict
    }


# ToDo: Logging of CRPS evaluation score
def ml_flow_tracking(data: dict, model: dict, hyperparameters_model: dict, hyperparameters_sampling: dict,
                     sampling_monitoring: dict, model_results: dict):
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
