# ToDo: How do we consider no. of knots in the model configuration (no_knots or no_coefficients)?
def fetch_model_run_config(model_name, data_specification: dict) -> dict:
    """
    Fetches the sampling and model parameters for a specific model.
    Args:
        model_name:
        data_specification: Information about the input data

    Returns:

    """
    stan_model_code = read_stan_model(model_name)
    # The configuration in the form of Python dictionaries
    config = {
        "sampling_parameters": {
            "num_samples": 1000,
            "num_tune": 1000,
            "target_accept": 0.95,
            "no_chains": 4,
            "no_ppc_samples": 500
        },
        "bayesian_psplines_gaussian_prior": {
            "model_name": model_name,
            "stan_model_code": stan_model_code,
            "prior_specification": """
                "intercept": "Normal",
                "slopesX1_0": "Normal",
                "slopesX1_rest": "Normal", 
                "tauX1": "InverseGamma",
                "slopesX2_0": "Normal",
                "slopesX2_rest": "Normal", 
                "tauX2": "InverseGamma",
                "likelihood": "NegativeBinomial",
                "alpha": "Gamma" 
            """,
            "model_hyperparameters_setting": {
                "spline_degree": 3,
                "num_knots_X1": {
                    "type": "int",
                    "range": [10, 40]
                },
                "num_knots_X2": {
                    "type": "int",
                    "range": [10, 40]
                },
                "a_tau_X1": {
                    "type": "float",
                    "range": [0.8, 1.2]
                },
                "b_tau_X1": {
                    "type": "float",
                    "range": [0.0045, 0.0055]
                },
                "a_tau_X2": {
                    "type": "float",
                    "range": [0.8, 1.2]
                },
                "b_tau_X2": {
                    "type": "float",
                    "range": [0.0045, 0.0055]
                },
                "a_alpha": {
                    "type": "float",
                    "range": [0.05, 0.15]
                },
                "b_alpha": {
                    "type": "float",
                    "range": [0.05, 0.15]
                },
                # ToDo: Consider dropping hyperparameter for intercept mu
                "intercept_mu": {
                    "type": "float",
                    "range": [-5, 5]
                },
                "intercept_sigma": {
                    "type": "float",
                    "range": [2, 8]
                },

            }
        }
    }
    # Check if model_name exists in the configuration
    if model_name not in config:
        raise ValueError(f"No configuration found for the model: {model_name}")

    # Fetch the sampling and model parameters
    stan_model = config[model_name]["stan_model_code"]
    sampling_parameters = config["sampling_parameters"]
    model_hyperparameters_setting = config[model_name]["model_hyperparameters_setting"]
    descritive_values = {
        "model_name": config[model_name]["model_name"],
        "prior_specification": config[model_name]["prior_specification"]
    return {
        "stan_model_code": stan_model,
        "sampling_parameters": sampling_parameters,
        "model_hyperparameters_setting": model_hyperparameters_setting,
        "descriptive_values": descritive_values,
        "data_specification": data_specification,
    }


def read_stan_model(model_name: str) -> str:
    # Get the stan model code from the directory
    model_file_path = f"StanModels/{model_name}.stan"
    with open(model_file_path, 'r') as file:
        stan_model_code = file.read()

    return stan_model_code
