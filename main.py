from data_gathering import gather_data_actuals, gather_data_features
from data_preparation import select_data, prepare_data
from data_visualization import basic_line_plot, import_country_mapping
from data_modelling_baseline import compute_baseline_tuning
from bayesian_regression import compute_bayesian_model
from help_functions import generate_datetime_list
from mappings import map_date_to_month_id

# STEP: DEFINE BOOLEAN FOR EXECUTION
# Define if line plot should be made
plot_lineplot_bool = True
# Define if data should be prepared
prepare_data_bool = True
# Define if baseline model should be run
run_baseline_model = False
# Define if bayesian model should be run
run_bayesian_model = True

# STEP: LOAD MAPPINGS
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
# ToDO: Needs to be revised
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
evaluation_years = [2015, 2016, 2017, 2018, 2019]

# STEP: Define parameters for baseline model
train_month_set_baseline = [6, 9, 12, 15, 18, 24, 30, 36]
forecast_horizon_set_baseline = [1, 2, 3, 4, 5, 6]
quantiles_set_baseline = [[0.1, 0.5, 0.9], [0.25, 0.5, 0.75], [0.1, 0.25, 0.5, 0.75, 0.9]]
output_directory_baseline = \
    r"C:\Users\Uwe Drauz\Documents\bachelor_thesis_local\personal_competition_data\Baseline_Model"

# STEP: Define parameters for bayesian model
# Note: if str: 'all' is passed, all available data should be used for training
train_month_set_bayesian = ['all']
forecast_horizon_set_bayesian = [1, 2, 3, 4, 5, 6]
sample_iterations_set_bayesian = [1000]
predictors = ['ged_sb_tlag_1', 'ged_sb_tsum_24']
target_variable = 'ged_sb'
bayesian_model_variables = ['month_id', 'country_id', target_variable] + predictors
data_bayesian_model = data_cm_features_allyears[bayesian_model_variables]
output_directory_bayesian = \
    r"C:\Users\Uwe Drauz\Documents\bachelor_thesis_local\personal_competition_data\Bayesian_Model"

for evaluation_year in evaluation_years:
    # Compute the CRPS scores averaged over (1) all countries and all forecasting months and (2) all forecasting months
    # Generate datetime objects corresponding to the evaluation_year
    # The forecast month set is no parameter to tune as for every month included we take a different training period
    # The training period and forecast horizon "moves" along with the forecasted months
    forecast_month_set = generate_datetime_list(start_year=evaluation_year, start_month=1,
                                                end_year=evaluation_year, end_month=12)
    first_month = map_date_to_month_id(forecast_month_set[0].year, forecast_month_set[0].month)
    last_month = map_date_to_month_id(forecast_month_set[-1].year, forecast_month_set[-1].month)

    # Extract country_ids of countries which do have data within the evaluation period
    country_ids = sorted(data_baseline_model[(first_month <= data_baseline_model['month_id']) &
                                             (data_baseline_model['month_id'] <= last_month)]['country_id'].unique())

    # STEP: Run Baseline Model
    if run_baseline_model is True:
        crps_scores_country_specific_baseline, crps_scores_global_baseline = compute_baseline_tuning(
            data=data_baseline_model, country_ids=country_ids, forecast_month_set=forecast_month_set,
            train_months_set=train_month_set_baseline, forecast_horizon_set=forecast_horizon_set_baseline,
            quantiles_set=quantiles_set_baseline, output_directory=output_directory_baseline, year=evaluation_year)

    # Note: The Bayesian model is not yet implemented:
    # Note: ZeroInflatedNegativeBinomial, sophisticated predictor implementation, P-splines, regularization
    # STEP: Run Bayesian Regression Model
    if run_bayesian_model is True:
        crps_scores_country_specific_bayesian, crps_scores_global_bayesian = compute_bayesian_model(
            data=data_bayesian_model, country_ids=country_ids, forecast_month_set=forecast_month_set,
            train_months_set=train_month_set_bayesian, forecast_horizon_set=forecast_horizon_set_bayesian,
            sample_iterations_set=sample_iterations_set_bayesian, predictors=predictors,
            target_variable=target_variable, output_directory=output_directory_bayesian, year=evaluation_year)

print('fin')
