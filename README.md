# Bachelor Thesis Repository

## Overview
This repository contains the data and Python code used for the bachelor's thesis "Bayesian Penalized Structured Regression Models for Probabilistic Forecasting of Conflict Fatalities"  in the context of the VIEWS 2023/2024 Prediction competition.
## Repository Structure
- **Data**: This directory includes various subdirectories containing actual data, codebooks, features, country lists, and neighbors lists, as well as prediction and score results.
- **Plots**: Contains plots for different sections of the thesis, including data section, result section with various types of visualizations like boxplots, heatmaps, line plots, etc.
- **Python Scripts and Notebooks**: Includes various Python scripts and Jupyter notebooks for processing and visualizing data, computing error metrics, and more.
### Data Directory
- **DateFromVIEWS**: Contains data extracted from the ViEWS database.
  - **Actuals**: Includes actual conflict fatality data.
  - **Codebooks**: Descriptions of the feature data used.
  - **Features**: Includes feature data.
  - **country_list**: A list of countries included in the study.
  - **country_neighbors**: Information about neighboring countries.
- **InputData**: This directory includes the processed data used as model input.
- **Plots**: Visualizations generated from the analysis.
  - **plots_for_data_section**: Plots related to the data exploration or preprocessing section of your thesis.
  - **plots_for_results_section**: Visualizations of the results, including various types of plots like boxplots, heatmaps, and line plots.
- **Results**: Contains the outcomes of the analyses.
  - **metrics**: Includes performance on scoring rules with all posterior predictive samples considered.
  - **metrics_with_thinning**: Includes performance on scoring rules with every 2nd posterior predictive samples considered.
  - **posterior_predictive_samples**: Stores samples from the posterior predictive distributions of models.
### StanModels Directory
- **StanModels**: Contains Stan models used in your Bayesian analysis. Stan is a programming language for statistical modeling. (not used in this repository)
### Key Files
- **baseline_configs.py**: Configuration settings for the baseline model.
- **Various Jupyter Notebooks**: These notebooks are used for tasks like visualizing data/ scores/ posterior samples, preparing data for Bayesian models, processing results, and computing error metrics.
- **data_gathering.py**: Script for gathering data.
- **data_preprocessing.py**: Script for preprocessing data.
- **data_visualization.py**: Script for visualizing data.
- **data_modeling.py**: Script for modeling data (baseline part used).
- **data_logging.py**: Script for logging data using mlflow.
- **helper_functions.py**: Script for helper functions.
- **mapping.py**: Script for mapping function.
- **main.py**: Script for running the whole pipeline.
### Other Files
- **requirements.txt**: Lists all Python dependencies required for the project (possibly some can be omitted).
## Additional Information
- The Bayesian model, while referenced in the Python scripts, was eventually implemented in [R in a separate reposertory](https://github.com/simondrauz/bachelor_thesis_RCode).
- In general the use of pystan on a Windows machine can be tricky. It did work using python 3.7 and installing the neede C++ compiler 