## Authors: Saksham Soni (sakshams@mit.edu)
import numpy as np
from .health_data.health_params import DAILY_HOSPITALIZATION_COST 

def hospitalization_costs(policy):
    region = policy.region
    total_hospitalized_days = policy.hospitalization_days
    total_icu_days = policy.icu_days
    total_ventilated_days = policy.ventilated_days
    
    inpatient_daily_cost = DAILY_HOSPITALIZATION_COST[region]["Inpatient"]
    inpatient_daily_cost = inpatient_daily_cost if inpatient_daily_cost is not None else 0
    icu_daily_cost = DAILY_HOSPITALIZATION_COST[region]["ICU bed"]
    icu_daily_cost = icu_daily_cost if icu_daily_cost is not None else 0
    ventilated_daily_cost = DAILY_HOSPITALIZATION_COST[region]["Ventilated ICU bed"]
    ventilated_daily_cost = ventilated_daily_cost if ventilated_daily_cost is not None else 0

    return icu_daily_cost*total_icu_days + ventilated_daily_cost*total_ventilated_days + inpatient_daily_cost*total_hospitalized_days
