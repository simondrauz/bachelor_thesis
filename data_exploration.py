
import os
import pandas as pd

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

print('fin')
