from datetime import datetime, timedelta


def generate_datetime_list(start_year: int, start_month: int, end_year: int, end_month: int) -> list:
    """
    Generates a list of datetime objects ranging from the specified starting year and month to the specified ending year and month.

    Args:
        start_year (int): The starting year.
        start_month (int): The starting month.
        end_year (int): The ending year.
        end_month (int): The ending month.

    Returns:
        list: A list of datetime objects representing the first day of each month within the specified range.
    """
    start_date = datetime(year=start_year, month=start_month, day=1)
    end_date = datetime(year=end_year, month=end_month, day=1)
    delta = timedelta(days=32)

    datetime_list = []
    current_date = start_date
    while current_date <= end_date:
        datetime_list.append(current_date)
        current_date += delta
        current_date = current_date.replace(day=1)

    return datetime_list
