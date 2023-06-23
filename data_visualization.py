import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from mappings import import_country_mapping


def get_country_name(country_id: int, country_mapping: pd.DataFrame) -> str:
    """
    Gets the country name corresponding to a data set if it only contains data of one country.
    :param df: data frame with specific country and time period
    :param country_mapping: mapping of country_id to country_name
    :return: country_name corresponding to data_set
    """

    if len(country_id) == 0:
        raise Exception('The given data frame contains no country_id/ is empty')
    elif len(country_id) == 1:
        country_name = country_mapping[country_mapping['country_id'] == country_id[0]]['name'].squeeze()

    else:
        raise Exception('The given data frame contains more than one country_id.')

    return country_name

# ToDo: Solve issues with export of plots (are displayed correctly in IDE, but white in file)
# ToDo: Countries with few conflict fatalities show x axes centered -->should be at the bottom
# ToDo: Automatically adjust Plot Title to time frame
# ToDo: Implement a mapping of month_id to actual month (probably more exhaustive task) -->ask for mapping of months
def basic_line_plot(df: pd.DataFrame, country_mapping: pd.DataFrame, start_month: int, sb_only=False, save=False):
    """
    Creates line graph plotting the different fatality counts against each other
    :param df: data frame with data of the defined country and time period
    :param country_mapping: mapping of country_id to country_name
    :param start_month: indicates the starting month of the time period
    :param sb_only: Set True if only plot of state_based conflicts
    :param save: Set true if plot should be saved
    :return:
    """
    # Extract the country corresponding to the data
    country_id = df['country_id'].unique()
    country_name = get_country_name(country_id, country_mapping)

    # Set the style of the plot
    sns.set(style="darkgrid")

    # Create the line plot
    sns.lineplot(data=df, x='month_id', y='ged_sb', label='state-based conflict')
    if sb_only is False:
        sns.lineplot(data=df, x='month_id', y='ged_ns', label='non-state-based conflict')
        sns.lineplot(data=df, x='month_id', y='ged_os', label='one-sided conflict')

    # Set the legend
    plt.legend(title='Variable')

    # Set the labels and title
    plt.xlabel('Month')
    plt.ylabel('Value')
    plt.title(f'Conflict Fatalities {country_name}')

    # Display the plot
    plt.show()

    # save the plot according to what they are displaying
    if save is True:
        if sb_only is True:
            if start_month == 121:
                savepath = rf'C:\Users\Uwe Drauz\Documents\bachelor_thesis_local\personal_competition_data\Plots\jan1990_oct2020_sb_lineplot\{country_id}_{country_name}_jan1990_oct2020.png'
                plt.savefig(savepath)
            elif start_month == 431:
                savepath = rf'C:\Users\Uwe Drauz\Documents\bachelor_thesis_local\personal_competition_data\Plots\oct2015_oct2020_sb_lineplot\{country_id}_{country_name}_oct2015_oct2020.png'
                plt.savefig(savepath)
            else:
                savepath = rf'C:\Users\Uwe Drauz\Documents\bachelor_thesis_local\personal_competition_data\Plots\other_time_period_sb_lineplot\{country_id}_{country_name}_start_month_{start_month}.png'
                plt.savefig(savepath)

        if sb_only is False:
            if start_month == 121:
                savepath = rf'C:\Users\Uwe Drauz\Documents\bachelor_thesis_local\personal_competition_data\Plots\jan1990_oct2020_sb_ns_os_lineplot\{country_id}_{country_name}_jan1990_oct2020.png'
                plt.savefig(savepath)
            elif start_month == 431:
                savepath = rf'C:\Users\Uwe Drauz\Documents\bachelor_thesis_local\personal_competition_data\Plots\oct2015_oct2020_sb_ns_os_lineplot\{country_id}_{country_name}_oct2015_oct2020.png'
                plt.savefig(savepath)
            else:
                savepath = rf'C:\Users\Uwe Drauz\Documents\bachelor_thesis_local\personal_competition_data\Plots\other_time_period_sb_ns_os_lineplot\{country_id}_{country_name}_start_month_{start_month}.png'
                plt.savefig(savepath)
