# STEP: IMPORT FUNCTIONS
import pandas as pd
from data_gathering import gather_data_features, gather_data_actuals
from data_exploration import calculate_correlation
# STEP: LOAD DATA
# Load actuals data
data_cm_actual_2018, data_cm_actual_2019, data_cm_actual_2020, data_cm_actual_2021, data_cm_actual_allyears \
    = gather_data_actuals()

# Load features data
data_cm_features_2017, data_cm_features_2018, data_cm_features_2019, data_cm_features_2020, data_cm_features_allyears \
    = gather_data_features()


output_directory = r"C:\Users\Uwe Drauz\Documents\bachelor_thesis_local\personal_competition_data\Baseline_Model"

crps_scores_all_year_country_specific = {}
crps_scores_all_year_global = {}
for year in [2015, 2016, 2017, 2018, 2019]:
    file_path = rf'C:\Users\Uwe Drauz\Documents\bachelor_thesis_local\personal_competition_data\Baseline_Model\crps_scores_all_year_{year}_country_specific.parquet'
    crps_scores_all_year_country_specific[year] = pd.read_parquet(file_path)
    file_path = rf'C:\Users\Uwe Drauz\Documents\bachelor_thesis_local\personal_competition_data\Baseline_Model\crps_scores_all_year_{year}_global.parquet'
    crps_scores_all_year_global[year] = pd.read_parquet(file_path)

crps_scores_all_year_country_specific_year_2015 = crps_scores_all_year_country_specific[2015]
crps_scores_all_year_country_specific_year_2016 = crps_scores_all_year_country_specific[2016]
crps_scores_all_year_country_specific_year_2017 = crps_scores_all_year_country_specific[2017]
crps_scores_all_year_country_specific_year_2018 = crps_scores_all_year_country_specific[2018]
crps_scores_all_year_country_specific_year_2019 = crps_scores_all_year_country_specific[2019]

crps_scores_all_year_global_year_2015 = crps_scores_all_year_global[2015]
crps_scores_all_year_global_year_2016 = crps_scores_all_year_global[2016]
crps_scores_all_year_global_year_2017 = crps_scores_all_year_global[2017]
crps_scores_all_year_global_year_2018 = crps_scores_all_year_global[2018]
crps_scores_all_year_global_year_2019 = crps_scores_all_year_global[2019]


# STEP: DATA EXPLORATION
input_variables = \
    data_cm_features_allyears.columns.drop(['index', 'country_id', 'month_id', 'ged_sb']).tolist()
target_variables = \
    ['ged_sb', 'ged_sb_tlag_1', 'ged_sb_tlag_2', 'ged_sb_tlag_3', 'ged_sb_tlag_4', 'ged_sb_tlag_5', 'ged_sb_tlag_6']
correlation_values = calculate_correlation(data_cm_features_allyears, target_variables=target_variables, input_variables=input_variables)

