from datetime import datetime

import pandas as pd


def import_country_mapping():
    country_list = pd.read_csv(
        r'C:\Users\Uwe Drauz\Documents\bachelor_thesis_local\shared_competition_data\country_list\country_list.csv')
    return country_list


def map_country_id_to_country_name(country_id: int) -> str:
    # Import country mapping
    country_mapping = import_country_mapping()
    # Determine country_name based upon entry in country_mapping with columns 'country_id' and 'name'
    country_name = country_mapping[country_mapping['country_id'] == country_id]['name'].values[0]
    return country_name


def map_country_name_to_country_id(country_name: str) -> int:
    # Import country mapping
    country_mapping = import_country_mapping()
    # Determine country_name based upon entry in country_mapping with columns 'country_id' and 'name'
    country_id = country_mapping[country_mapping['name'] == country_name]['country_id'].values[0]
    return country_id


def map_date_to_month_id(year: int, month: int) -> int:
    """
    Maps a datetime object to the corresponding month_id number.

    Args:
        year (int): year
        month (int): month
    Returns:
        int: The corresponding month_id number.

    """
    date = datetime(year=year, month=month, day=1)
    base_date = datetime(1990, 1, 1)
    months_diff = (date.year - base_date.year) * 12 + (date.month - base_date.month)
    month_id = 121 + months_diff
    return month_id


def map_month_id_to_datetime(month_id: int) -> datetime:
    """
    Maps a month_id to a datetime object with day=1.

    Args:
        month_id (int): The input month_id.

    Returns:
        datetime: The corresponding datetime object with day=1.
    """
    base_date = datetime(1990, 1, 1)
    months_diff = month_id - 121
    years_diff = months_diff // 12
    months_remainder = months_diff % 12

    target_year = base_date.year + years_diff
    target_month = base_date.month + months_remainder
    if target_month > 12:
        target_year += 1
        target_month -= 12

    target_date = datetime(target_year, target_month, 1)
    return target_date
