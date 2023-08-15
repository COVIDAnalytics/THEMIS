from datetime import datetime
import pandas as pd

# 2016 global population data
global_populations = pd.read_csv("pandemic_functions/pandemic_data/Population_Global.csv")
# oxford policy data
# raw_measures = pd.read_csv("https://github.com/OxCGRT/covid-policy-tracker/raw/master/data/OxCGRT_nat_latest.csv")
raw_measures = pd.read_csv("raw_data/OxCGRT_nat_latest.csv")
# raw policy data for US
df_raw_us_policies = pd.read_csv("pandemic_functions/pandemic_data/12062020_raw_policy_data_us_only.csv")

default_parameter_list = [1, 0, 2, 0.2, 0.05, 0.2, 3, 3, 0.1, 3, 1] # Default parameters for the solver

### Parameters to control the model fitting process
dict_default_reinit_parameters = {
    "alpha": 0, "days": None, "r_s": 0, "r_dth": 0.02, "p_dth": 0, "r_dthdecay": 0,
    "k1": 0, "k2": 0, "jump": 0, "t_jump": 0, "std_normal": 1,
}  # Allows for reinitialization of parameters in case they reach a value that is too low/high
dict_default_reinit_lower_bounds = {
    "alpha": 0, "days": None, "r_s": 0, "r_dth": 0.02, "p_dth": 0, "r_dthdecay": 0,
    "k1": 0, "k2": 0, "jump": 0, "t_jump": 0, "std_normal": 1,
}  # Allows for reinitialization of lower bounds in case they reach a value that is too low
dict_default_reinit_upper_bounds = {
    "alpha": 0, "days": None, "r_s": 0, "r_dth": 0.02, "p_dth": 0, "r_dthdecay": 0,
    "k1": 0, "k2": 0, "jump": 0, "t_jump": 0, "std_normal": 1,
}  # Allows for reinitialization of upper bounds in case they reach a value that is too high

# Deafault parameters - TNC
default_upper_bound = 0.2
percentage_drift_upper_bound = 0.2
default_lower_bound = 0.2
percentage_drift_lower_bound = 0.2
default_bounds_params = (
    (0.75, 1.25), (-10, 10), (1, 3), (0.05, 0.5), (0.01, 0.25), (0, 0.5), (0.1, 10), (0.1, 10), (0, 5), (0, 7), (0.1, 5)
)  # Bounds for the solver

validcases_threshold = 7  # Minimum number of cases to fit the base-DELPHI
validcases_threshold_policy = 15  # Minimum number of cases to train the country-level policy predictions
max_iter = 500  # Maximum number of iterations for the algorithm

# Default parameters - Annealing
percentage_drift_upper_bound_annealing = 1
default_upper_bound_annealing = 1
percentage_drift_lower_bound_annealing = 1
default_lower_bound_annealing = 1
default_lower_bound_jump = 0
default_upper_bound_jump = 5
default_lower_bound_std_normal = 1
default_upper_bound_std_normal = 100

default_maxT = datetime(2020, 12, 31)  # Maximum timespan of prediction
n_params_without_policy_params = 7  # alpha, r_dth, p_dth, a, b, k1, k2

region_symbol_country_dict = {
    "DE": ("Germany", "None"),
    "US": ("US", "None"),
    "FR": ("France", "None"),
    "US-NY": ("US","New York"),
    "US-FL": ("US", "Florida"),
    "ES": ("Spain", "None"),
    "BR": ("Brazil", "None"),
    "SG": ("Singapore", "None")
}

region_symbol_continent_dict = {
    "DE": "Europe",
    "US": "North America",
    "FR": "Europe",
    "US-NY": "North America",
    "US-FL": "North America",
    "ES": "North America",
    "BR": "South America",
    "SG": "Asia"
}

### DELPHI parameters for poilicy scenarios

validcases_threshold_policy = 15

"""
Fixed Parameters based on meta-analysis:
p_h: Hospitalization Percentage
RecoverHD: Average Days until Recovery
VentilationD: Number of Days on Ventilation for Ventilated Patients
maxT: Maximum # of Days Modeled
p_d: Percentage of True Cases Detected
p_v: Percentage of Hospitalized Patients Ventilated
"""
p_d=0.2 # Probability of detection
p_h=0.15 # Probability of hospitalization
p_v = 0.25  # Percentage of ventilated
IncubeD = 5
RecoverID = 10
RecoverHD = 15
DetectD = 2
VentilatedD = 10  # Recovery Time when Ventilated

policy_data_start_date = "2020-03-01"
policy_data_end_date = "2020-07-31"

future_policies = [
    'No_Measure', 'Restrict_Mass_Gatherings', 'Mass_Gatherings_Authorized_But_Others_Restricted',
    'Restrict_Mass_Gatherings_and_Schools', 'Authorize_Schools_but_Restrict_Mass_Gatherings_and_Others',
    'Restrict_Mass_Gatherings_and_Schools_and_Others', 'Lockdown'
]

# Default normalized gamma shifts from runs in May 2020
default_dict_normalized_policy_gamma = {
    'No_Measure': 1.0,
    'Restrict_Mass_Gatherings': 0.873,
    'Authorize_Schools_but_Restrict_Mass_Gatherings_and_Others': 0.794,
    'Mass_Gatherings_Authorized_But_Others_Restricted': 0.668,
    'Restrict_Mass_Gatherings_and_Schools': 0.479,
    'Restrict_Mass_Gatherings_and_Schools_and_Others': 0.423,
    'Lockdown': 0.239
}