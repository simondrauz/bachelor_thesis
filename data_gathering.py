import pandas as pd


def gather_data_actuals():
    """
    Load the actuals dataframes for each year and all years.

    Returns:
        Tuple: Dataframes containing the actuals data for each year and all years.
    """
    data_cm_actual_2018 = pd.read_parquet(
        r"C:\Users\Uwe Drauz\Documents\bachelor_thesis_local\shared_competition_data\Actuals\cm_actuals_2018.parquet")
    data_cm_actual_2018 = data_cm_actual_2018.reset_index()

    data_cm_actual_2019 = pd.read_parquet(
        r"C:\Users\Uwe Drauz\Documents\bachelor_thesis_local\shared_competition_data\Actuals\cm_actuals_2019.parquet")
    data_cm_actual_2019 = data_cm_actual_2019.reset_index()

    data_cm_actual_2020 = pd.read_parquet(
        r"C:\Users\Uwe Drauz\Documents\bachelor_thesis_local\shared_competition_data\Actuals\cm_actuals_2020.parquet")
    data_cm_actual_2020 = data_cm_actual_2020.reset_index()

    data_cm_actual_2021 = pd.read_parquet(
        r"C:\Users\Uwe Drauz\Documents\bachelor_thesis_local\shared_competition_data\Actuals\cm_actuals_2021.parquet")
    data_cm_actual_2021 = data_cm_actual_2021.reset_index()

    data_cm_actual_allyears = pd.read_parquet(
        r"C:\Users\Uwe Drauz\Documents\bachelor_thesis_local\shared_competition_data\Actuals\cm_actuals_allyears.parquet")
    data_cm_actual_allyears = data_cm_actual_allyears.reset_index()

    return (data_cm_actual_2018, data_cm_actual_2019, data_cm_actual_2020, data_cm_actual_2021,
            data_cm_actual_allyears)


def gather_data_features():
    """
    Load the features dataframes for each year and all years.

    Returns:
        Tuple: Dataframes containing the features data for each year and all years.
    """
    data_cm_features_2017 = pd.read_parquet(
        r"C:\Users\Uwe Drauz\Documents\bachelor_thesis_local\shared_competition_data\Features\cm_features_to_oct2017.parquet")
    data_cm_features_2017 = data_cm_features_2017.reset_index()

    data_cm_features_2018 = pd.read_parquet(
        r"C:\Users\Uwe Drauz\Documents\bachelor_thesis_local\shared_competition_data\Features\cm_features_to_oct2018.parquet")
    data_cm_features_2018 = data_cm_features_2018.reset_index()

    data_cm_features_2019 = pd.read_parquet(
        r"C:\Users\Uwe Drauz\Documents\bachelor_thesis_local\shared_competition_data\Features\cm_features_to_oct2019.parquet")
    data_cm_features_2019 = data_cm_features_2019.reset_index()

    data_cm_features_2020 = pd.read_parquet(
        r"C:\Users\Uwe Drauz\Documents\bachelor_thesis_local\shared_competition_data\Features\cm_features_to_oct2020.parquet")
    data_cm_features_2020 = data_cm_features_2020.reset_index()

    data_cm_features_allyears = pd.concat(
        [data_cm_features_2017, data_cm_features_2018, data_cm_features_2019, data_cm_features_2020])
    data_cm_features_allyears = data_cm_features_allyears.reset_index()

    return (data_cm_features_2017, data_cm_features_2018, data_cm_features_2019, data_cm_features_2020,
            data_cm_features_allyears)
