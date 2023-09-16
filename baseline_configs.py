def fetch_baseline_model_run_config(data_specification: dict) -> dict:
    """
    Fetches the baseline model parameters for the deterministic model.
    Args:
        data_specification: Information about the input data
    Returns:
        dict: Configuration settings for the baseline model
    """

# ToDo: Consider range approach for train_month_set and forecast_horizon_set
    config = {
        "cross_validation_settings": {
            "k_fold_iterations": 10,
            "cross_validation_type": "rolling_window",
            "forecast_horizon": 3,
            "target_variable": "ged_sb",
        },
        "model_hyperparameter_settings": {
            "rolling_window_length": {
                "type": "int",
                "range": [6, 36],
                # Discrete values alternativly: [6, 9, 12, 15, 18, 24, 30, 36]
            },
            "quantiles": {
                "n_quantiles": {
                    "type": "int",
                    "range": [2, 9]
                },
                "quantile_value": {
                    "type": "float",
                    "range": [0.05, 0.95],
                    "step": 0.05
                }
            },
            # Discrete values alternativly and selection with optuna categorical
            # "quantiles_set": [
            #     [0.1, 0.5, 0.9],
            #     [0.25, 0.5, 0.75],
            #     [0.1, 0.25, 0.5, 0.75, 0.9]
            # ],  # Different quantiles to estimate
        },
        "descriptive_values": {
            "model_name": "BaselineModel",
        }
    }

    return {
        "cross_validation_settings": config["cross_validation_settings"],
        "model_hyperparameter_settings": config["model_hyperparameter_settings"],
        "data_specification": data_specification,
        "descriptive_values": config["descriptive_values"],
    }
