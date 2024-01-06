import time

from baseline_configs import fetch_baseline_model_run_config
from bayesian_models_configs import fetch_model_run_config
from data_gathering import gather_data_actuals, gather_data_features
from data_modelling import run_model
from data_preparation import preprocess_data
from data_preparation import select_data, prepare_data, generate_data_dict_pre_modelling
from data_visualization import basic_line_plot, import_country_mapping
from help_functions import MLFlowServer, seconds_to_format

if __name__ == '__main__':
    # STEP: DEFINE BOOLEAN BEFORE EXECUTION
    # Define if line plot should be made
    plot_lineplot_bool = False
    # Define if data should be prepared
    prepare_data_bool = False
    # Define if baseline model should be run
    run_baseline_model_bool = False
    # Define if bayesian model should be run
    run_bayesian_model_bool = True
    # Define if mlflow server should be started
    start_mlflow_server_bool = False

    # ToDo: Remove setting model_experiment_name, if still not needed
    # STEP: SET NAME OF MODEL EXPERIMENT AND START MLFLOW SERVER
    if start_mlflow_server_bool is True:
        if run_bayesian_model_bool is True:
            model_experiment_name = "bayesian_models"
        else: model_experiment_name = "baseline_model"

        # Start MLFlow server
        server = MLFlowServer()
        server.start()


    # STEP: LOAD MAPPINGS
    # ToDo: Consider moving this into the model function
    # Get country_mapping
    country_mapping = import_country_mapping()

    # STEP: LOAD DATA
    # Load 'Actuals' data
    data_cm_actual_2018, data_cm_actual_2019, data_cm_actual_2020, data_cm_actual_2021, data_cm_actual_allyears \
        = gather_data_actuals()

    # Load features data
    data_cm_features_2017, data_cm_features_2018, data_cm_features_2019, data_cm_features_2020, data_cm_features_allyears \
        = gather_data_features()

    data_baseline_model = data_cm_features_allyears[['month_id', 'country_id', 'ged_sb']]
    # STEP: PREPARE DATA
    # ToDO: Adjust this step such that it calls the data precessing for the whole data frame
    if prepare_data_bool is True:
        data_baseline_model = prepare_data(data_baseline_model, variables_to_check=['ged_sb'])

    # STEP: PLOT DATA
    if plot_lineplot_bool is True:
        time_periods = [431, 121]
        for country_id in country_mapping['country_id']:
            for month in time_periods:
                # Select spatial and temporal data
                data = select_data(df=data_cm_features_allyears, country_id=country_id, month_cut=month)
                # Plot the data
                basic_line_plot(data, country_mapping=country_mapping, start_month=month, sb_only=True, save=True)

    # Year to be evaluated
    evaluation_years = [2018, 2019, 2020, 2021]
    dict_actuals = {
        "2018": data_cm_actual_2018,
        "2019": data_cm_actual_2019,
        "2020": data_cm_actual_2020,
        "2021": data_cm_actual_2021
        }

    # Determine all countries
    actual_countries = data_cm_actual_allyears['country_id'].unique()
    # Determine countries with at least one conflict fatality
    feature_countries_non_zero = data_cm_features_allyears[data_cm_features_allyears['ged_sb'] > 0][
        'country_id'].unique()
    # Determine countries with at least one conflict fatality and in actuals data
    feature_and_actuals_countries_non_zero = list(set(feature_countries_non_zero) & set(actual_countries))
    if run_bayesian_model_bool is True:
        # ToDo: Decide on which countries the bayesian model should be trained
        # STEP: FEATURE SELECTION AND STANDARDIZATION
        covariates = []
        lagged_covariates = ['ged_sb', 'ged_sb_tsum_24', 'decay_ged_sb_5', 'decay_ged_sb_100', 'decay_ged_sb_500', 'wdi_sp_pop_totl']
        # Select countries which have at least one conflict fatality
        data_cm_features_allyears = data_cm_features_allyears[data_cm_features_allyears['country_id'].isin(feature_and_actuals_countries_non_zero)]
        data_cm_features_allyears = preprocess_data(data_cm_features_allyears, covariates, lagged_covariates,
                                                    standardize=True, lags_needed=int, drop_na=True)
    if run_baseline_model_bool is True:
        # STEP: Experimental filtering on countries with conflict fatalities
        # Filter feature data to only include countries with at least one conflict fatality and also in the actuals data
        data_cm_features_allyears = data_cm_features_allyears[
            data_cm_features_allyears['country_id'].isin(feature_and_actuals_countries_non_zero)]

        # data_cm_features_allyears = data_cm_features_allyears[(data_cm_features_allyears['country_id'].isin(
        #     countries_ids_no_zero) & data_cm_features_allyears['country_id'].isin(countries_ids_all_actuals))]
        # filter actuals data to only include countries with at least one conflict fatality
        for evaluation_year in evaluation_years:
            dict_actuals[str(evaluation_year)] = dict_actuals[str(evaluation_year)][
                dict_actuals[str(evaluation_year)]['country_id'].isin(feature_countries_non_zero)]

        print("Running baseline model on countries with at least one conflict fatality")
        print(f"Number of countries with at least one conflict fatality: {len(feature_and_actuals_countries_non_zero)} of "
              f"{len(actual_countries)} countries in actuals data")


    # STEP: Generate data specification
    predictors = ['ged_sb_tlag_1', 'ged_sb_tsum_24']
    target_variable = 'ged_sb'
    data_transformation = "target variable not standardized, input data standardized"
    data_specification = generate_data_dict_pre_modelling(data_transformation, predictors, target_variable)
    # STEP: Get configuration for model run
    if run_baseline_model_bool is True:
        model_name = "BaselineModel"
        model_run_config = fetch_baseline_model_run_config(data_specification)
    if run_bayesian_model_bool is True:
        model_name = "bayesian_psplines_gaussian_prior_integrated_functionalities"
        model_run_config = fetch_model_run_config(model_name, data_specification)
    # ToDo: Implement preprocessing of data
    for evaluation_year in evaluation_years:
        actuals = dict_actuals[str(evaluation_year)]
        start_time = time.time()
        if run_baseline_model_bool:
            actual_test_results_global, test_results_global, test_results_all_countries, competition_contribution_results = (
                run_model(evaluation_year, model_name, actuals=actuals, training_data=data_cm_features_allyears,
                          model_run_config=model_run_config, optuna_trials=36, validate_on_horizon=True, validate_on_horizon_and_month=False,
                          is_bayesian_model=run_bayesian_model_bool, is_baseline_model=run_baseline_model_bool,
                          country_mapping=country_mapping))
        elif run_bayesian_model_bool:
            actuals_results, hyperparameter_tuning_results = (run_model(evaluation_year, model_name, actuals=actuals, training_data=data_cm_features_allyears,
                          model_run_config=model_run_config, optuna_trials=36, validate_on_horizon=True, validate_on_horizon_and_month=False,
                          is_bayesian_model=run_bayesian_model_bool, is_baseline_model=run_baseline_model_bool,
                          country_mapping=country_mapping))
        end_time = time.time() - start_time
        print(f"Model run for {evaluation_year} with {model_name} took {seconds_to_format(end_time)} seconds while "
              f"validating on horizon." )

        start_time = time.time()
        if run_baseline_model_bool:
            actual_test_results_global, test_results_global, test_results_all_countries, competition_contribution_results = (
                run_model(evaluation_year, model_name, actuals=actuals, training_data=data_cm_features_allyears,
                          model_run_config=model_run_config, optuna_trials=36, validate_on_horizon=False,
                          validate_on_horizon_and_month=True,
                          is_bayesian_model=run_bayesian_model_bool, is_baseline_model=run_baseline_model_bool,
                          country_mapping=country_mapping))
        elif run_bayesian_model_bool:
            actuals_results, hyperparameter_tuning_results = (
                run_model(evaluation_year, model_name, actuals=actuals, training_data=data_cm_features_allyears,
                          model_run_config=model_run_config, optuna_trials=36, validate_on_horizon=False,
                          validate_on_horizon_and_month=True,
                          is_bayesian_model=run_bayesian_model_bool, is_baseline_model=run_baseline_model_bool,
                          country_mapping=country_mapping))
        end_time = time.time() - start_time
        print(f"Model run for {evaluation_year} with {model_name} took {seconds_to_format(end_time)} seconds while "
              f"validating on horizon and month.")
