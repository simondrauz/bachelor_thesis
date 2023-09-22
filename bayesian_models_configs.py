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
        "cross_validation_settings": {
            "k_fold_iterations": 10,
            # ToDo: Consider leaving out cross-validation_type
            "cross-validation_type": "rolling_window",
            "forecast_horizon": 3,
            "posterior_predictive": ["y_pred_train", "y_pred_eval"]
        },
        "sampling_parameters": {
            'iter': 3000,  # 2000 tuning iterations + 1000 sampling iterations
            'warmup': 2000,  # Number of tuning iterations
            'chains': 4,  # Number of chains
            'seed': 12345,  # Random seed
            'control': {'adapt_delta': 0.95}  # Target acceptance rate
        },
        "bayesian_psplines_gaussian_prior_integrated_functionalities": {
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
            "data_generation_specification": {
                # Note: if this is changed to dynamic, it has to be considered in logging mlflow cal because
                #       mlflow.params doesn't expect nested dictionaries
                # We change the inputs dynamically with the forecast horizon to use the most recent data as input
                "covariates": {
                    "general_covariates": [],
                    "horizon_specific_covariates": {
                        "forecast_horizon_3": ['ged_sb_tlag_3', 'ged_sb_tsum_24_tlag_3'],
                        "forecast_horizon_4": ['ged_sb_tlag_4', 'ged_sb_tsum_24_tlag_4'],
                        "forecast_horizon_5": ['ged_sb_tlag_5', 'ged_sb_tsum_24_tlag_5'],
                        "forecast_horizon_6": ['ged_sb_tlag_6', 'ged_sb_tsum_24_tlag_6'],
                        "forecast_horizon_7": ['ged_sb_tlag_7', 'ged_sb_tsum_24_tlag_7'],
                        "forecast_horizon_8": ['ged_sb_tlag_8', 'ged_sb_tsum_24_tlag_8'],
                        "forecast_horizon_9": ['ged_sb_tlag_9', 'ged_sb_tsum_24_tlag_9'],
                        "forecast_horizon_10": ['ged_sb_tlag_10', 'ged_sb_tsum_24_tlag_10'],
                        "forecast_horizon_11": ['ged_sb_tlag_11', 'ged_sb_tsum_24_tlag_11'],
                        "forecast_horizon_12": ['ged_sb_tlag_12', 'ged_sb_tsum_24_tlag_12'],
                        "forecast_horizon_13": ['ged_sb_tlag_13', 'ged_sb_tsum_24_tlag_13'],
                        "forecast_horizon_14": ['ged_sb_tlag_14', 'ged_sb_tsum_24_tlag_14'],
                    }
                },
                "target_variable": "ged_sb",
                "spline_degree": 3,
                "random_walk_order": {
                    "random_walk_order_X1": 1,
                    "random_walk_order_X2": 1
                }
                # OPTIONAL: Consider adding specification of penalty order here
            },
            "model_hyperparameter_settings": {
                # number of interior knots
                "no_interior_knots": {
                    "no_interior_knots_X1": {
                        "type": "int",
                        "range": [10, 40]
                    },
                    "no_interior_knots_X2": {
                        "type": "int",
                        "range": [10, 40]
                    }
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
    cross_validation_settings = config["cross_validation_settings"]
    stan_model = config[model_name]["stan_model_code"]
    sampling_parameters = config["sampling_parameters"]
    data_generation_specification = config[model_name]["data_generation_specification"]
    model_hyperparameter_settings = config[model_name]["model_hyperparameter_settings"]
    descritive_values = {
        "model_name": config[model_name]["model_name"],
        "prior_specification": config[model_name]["prior_specification"]
    }
    return {
        "cross_validation_settings": cross_validation_settings,
        "stan_model_code": stan_model,
        "sampling_parameters": sampling_parameters,
        "data_generation_specification": data_generation_specification,
        "model_hyperparameter_settings": model_hyperparameter_settings,
        "descriptive_values": descritive_values,
        "data_specification": data_specification,
    }


# ToDo: Adjust the storing and reading of the model so functions and rest are separated
def read_stan_model(model_name: str) -> str:
    # Get the stan model code from the directory
    model_file_path = f"StanModels/{model_name}.stan"
    with open(model_file_path, 'r') as file:
        stan_model_code = file.read()

    return stan_model_code
