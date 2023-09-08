import CRPS.CRPS as pscore
import arviz as az
import numpy as np
import pandas as pd


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
