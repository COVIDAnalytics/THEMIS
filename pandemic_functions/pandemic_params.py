
from datetime import datetime

region_symbol_country_dict = {
    "GM": "Germany",
    "US": "US",
    "FR": "France"
}

### DELPHI parameters for poilicy scenarios

default_maxT_policies = datetime(2020,12,31)

p_d=0.2 # Probability of detection
p_h=0.15 # Probability of hospitalization
p_v = 0.25  # Percentage of ventilated
IncubeD = 5
RecoverID = 10
RecoverHD = 15
DetectD = 2
VentilatedD = 10  # Recovery Time when Ventilated

# Policies and future times for counterfactual predictions
future_policies = [
    'No_Measure', 'Restrict_Mass_Gatherings', 'Mass_Gatherings_Authorized_But_Others_Restricted',
    'Restrict_Mass_Gatherings_and_Schools', 'Authorize_Schools_but_Restrict_Mass_Gatherings_and_Others',
    'Restrict_Mass_Gatherings_and_Schools_and_Others', 'Lockdown'
]
future_times = [0, 7, 14, 28, 42]

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