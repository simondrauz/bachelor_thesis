import json
import os

import CRPS.CRPS as pscore
import arviz as az
import mlflow
import numpy as np
import pandas as pd
from mlflow.data.pandas_dataset import PandasDataset

from help_functions import seconds_to_format


def calculate_crps(idata: az.InferenceData, actuals: pd.DataFrame) -> (pd.DataFrame, float):
    """
    Calculate the Continuous Ranked Probability Score (CRPS) of a Bayesian model for each observation and the average CRPS.

    Parameters:
        idata (az.InferenceData): InferenceData object containing the posterior predictive.
        actuals (pd.DataFrame): DataFrame containing the observed data. Each row corresponds to an observation.

    Returns:
        crps_scores (pd.DataFrame): A Series containing the CRPS score for each observation.
        average_crps (float): The average CRPS score across all observations.
    """
    try:
        # Access the posterior predictive samples for each observation
        posterior_samples = idata['posterior_predictive']['obs'].values
    except KeyError:
        raise KeyError("The InferenceData object does not contain posterior predictive samples.")

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


def generate_data_dict(training_data: pd.DataFrame, data_characteristics: dict, data_transformation: str,

                       country_name: str, predictors: list) -> dict:
    """
    Information about data used in model.
    Args:
        training_data:
        data_characteristics: dictionary containing information about training data
        data_transformation: information about data transformation
        country_name:
        predictors:

    Returns:

    """
    # Note: pycharm highlighting doesn't recognize function, but isolated check shows that it works
    training_data_mlflow: PandasDataset = mlflow.data.from_pandas(df=training_data)
    return {
        "training_data": training_data_mlflow,
        "data_characteristics": data_characteristics,
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


def generate_model_hyperparameters_dict(predictors: list,
                                        no_knots_per_regressor: list[int],
                                        tau_hyperparameters: list[tuple[float, float]],
                                        alpha_nb_hyperparameters: tuple[float, float],
                                        intercept_hyperparameters: tuple[float, float]) -> dict:
    """
    Information about model hyperparameters.
    Args:
        predictors: list of predictors used in model for clear assignment of hyperparameters to predictors
        no_knots_per_regressor: no. of knots corresponding to each predictor in list 'predictors', as list with each
                                element being an integer corresponding to a predictor
        tau_hyperparameters: alpha and beta hyperparameters for scaling factor tau corresponding to each predictor in
                             list 'predictors', as list with each element being a tuple of alpha and beta
                             hyperparameters corresponding to a predictor
        alpha_nb_hyperparameters: alpha and beta hyperparameters for Gamma distribution of alpha parameter of
                                  negative binomial distribution
        intercept_hyperparameters: mu and sigma for normal distribution of intercept

    Returns:

    """
    # Check that the length of hyperparameter inputs connected to predictors is the same as the length of predictors
    assert len(predictors) == len(
        no_knots_per_regressor), "Mismatch in length between predictors and no_knots_per_regressor"
    assert len(predictors) == len(tau_hyperparameters), "Mismatch in length between predictors and tau_hyperparameters"

    return {
        # Create dictionaries for some parameters that they can be clearly assigned to a predictor
        "no_knots_per_regressor": {
            predictors[i]: no_knots_per_regressor[i] for i in range(len(predictors))
        },
        "tau_hyperparameters": {
            predictors[i]: tau_hyperparameters[i] for i in range(len(predictors))
        },
        "alpha_nb_hyperparameters": alpha_nb_hyperparameters,
        "intercept_hyperparameters": intercept_hyperparameters
    }


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


def generate_model_results_dict(idata: az.InferenceData, crps_score: float) -> dict:
    """
    Information about model results.
    Args:
        idata: InferenceData object containing posterior samples, sample iterations, posterior probabilities
        crps_score: average crps score of model over validation set

    Returns:

    """
    return {
        "idata": idata,
        "crps_score": crps_score
    }


def generate_all_dictionaries_pymc(training_data: pd.DataFrame, data_characteristics: dict, data_transformation: str,
                                   country_name: str, predictors: list,
                                   model_name: str, prior_specifications: dict, no_knots_per_regressor: list[int],
                                   tau_hyperparameters: list[tuple[float, float]],
                                   alpha_nb_hyperparameters: tuple[float, float],
                                   intercept_hyperparameters: tuple[float, float], tuning_iterations: int,
                                   sampling_iterations: int, target_acceptance_rate: float, chains: int,
                                   divergences: int, sampling_time: float, idata: az.InferenceData,
                                   crps_score: float) -> dict:
    data_dict = generate_data_dict(training_data, data_characteristics, data_transformation, country_name, predictors)
    model_dict = generate_model_dict(model_name, prior_specifications)
    model_hyperparameters_dict = generate_model_hyperparameters_dict(predictors, no_knots_per_regressor,
                                                                     tau_hyperparameters,
                                                                     alpha_nb_hyperparameters,
                                                                     intercept_hyperparameters)
    sampling_hyperparameters_dict = generate_sampling_hyperparameters_dict(tuning_iterations, sampling_iterations,
                                                                           target_acceptance_rate, chains)
    sampling_monitoring_dict = generate_sampling_monitoring_dict(divergences, sampling_time)
    model_results_dict = generate_model_results_dict(idata, crps_score)

    return {
        "data": data_dict,
        "model": model_dict,
        "hyperparameters_model": model_hyperparameters_dict,
        "hyperparameters_sampling": sampling_hyperparameters_dict,
        "sampling_monitoring": sampling_monitoring_dict,
        "model_results": model_results_dict
    }


def ml_flow_tracking_pymc(data: dict, model: dict, hyperparameters_model: dict, hyperparameters_sampling: dict,
                          sampling_monitoring: dict, model_results: dict):
    """
    Preliminary function to track hyperparameters and results of pymc model
    Args:
        data: training data, data characteristics, data transformation, country name, predictors
        model: model description, prior specifications
        hyperparameters_model: no. of knots per regressor, tau hyperparameters, alpha hyperparameters,
                               intercept hyperparameters
        hyperparameters_sampling: tuning iterations, sampling iterations, target acceptance rate, chains
        sampling_monitoring: divergences, sampling time
        model_results: InferenceData, crps score

    Returns:

    """
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    with mlflow.start_run(run_name=model["model_name"]):
        # Log data parameters
        for key, value in data.items():
            if key == 'training_data':
                mlflow.log_input(value, context="training")
            elif key == 'data_characteristics':
                for char_key, char_value in value.items():
                    # Convert the dictionary to a JSON string
                    serialized_char_value = json.dumps(char_value)

                    mlflow.log_param(char_key, serialized_char_value)
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

        # Log CRPS score
        mlflow.log_metric('crps', model_results["crps_score"])

        # Convert InferenceData to PandasDataframe and log it with ml.data
        try:
            idata_df = model_results["idata"].to_dataframe()
            idata_df_mlflow: PandasDataset = mlflow.data.from_pandas(df=idata_df)
            mlflow.log_input(idata_df_mlflow, context="inference_data")
        except Exception as e:
            print(f"Error logging InferenceData as PandasDataframe: {e}")

        # Store the artifact in the artifact folder connected to the current run
        # Get the current run's artifact URI
        artifact_dir = os.path.join(os.getcwd(), "mlruns", "0", mlflow.active_run().info.run_id, "artifacts")
        # Export and log the InferenceData as an artifact
        try:
            # Logs only InferenceData object, not the whole artifact folder
            idata_path = os.path.join(artifact_dir, f"{model['model_name']}_results.json")
            model_results["idata"].to_json(idata_path)
            mlflow.log_artifact(idata_path)
        except Exception as e:
            print(f"Error logging InferenceData as an artifact: {e}")
