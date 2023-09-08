from datetime import datetime, timedelta
import subprocess
import time
import mlflow
import socket


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


def seconds_to_format(seconds: float) -> str:
    # Convert seconds to integer
    int_seconds = int(seconds)

    # Calculate hours, minutes, and remaining seconds
    hours = int_seconds // 3600
    int_seconds %= 3600
    minutes = int_seconds // 60
    remaining_seconds = int_seconds % 60

    # Return in the desired format
    return f"{hours}h {minutes}min {remaining_seconds}s"


class MLFlowServer:
    def __init__(self, experiment_name: str, tracking_uri="http://127.0.0.1:5000"):
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.process = None

    def _is_port_in_use(self, port: int) -> bool:
        """
        Checks if the given port is already in use.
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) == 0

    def start(self):
        """
        Starts the MLflow user interface and sets the tracking URI and experiment.
        Only starts if the port is not already in use.
        """
        if not self._is_port_in_use(5000):
            self.process = subprocess.Popen(["mlflow", "ui"])
            time.sleep(5)  # give it some time to start
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(self.experiment_name)
        else:
            print("MLflow server is already running or port 5000 is in use.")

# FIxMe: The stop function does not stop the server, consider leaving it running
    def stop(self):
        """
        Shuts down the MLflow server.
        """
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)  # Wait for 5 seconds to see if the process terminates
            except subprocess.TimeoutExpired:
                print("MLflow server did not terminate gracefully. Forcibly killing.")
                self.process.kill()
            self.process = None