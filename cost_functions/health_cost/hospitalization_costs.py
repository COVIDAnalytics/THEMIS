## Authors: Saksham Soni (sakshams@mit.edu)
import numpy as np
from cost_functions.health_cost.health_data.health_params import DAILY_HOSPITALIZATION_COST 

def hospitalization_costs(pandemic):
    """
    Function that returns the cost of hospitalizations due to pandemic in a region
    Parameters:
        - pandemic: Pandemic object containing the information of the region and duration that is being analyzed
    Returns:
        - Total hospitalization costs
    """
    region = pandemic.region
    total_hospitalized_days =np.array([pandemic.hospitalization_days, pandemic.hospitalization_days_lb, pandemic.hospitalization_days_ub])
    total_icu_days = np.array([pandemic.icu_days, pandemic.icu_days_lb, pandemic.icu_days_ub])
    total_ventilated_days = np.array([pandemic.ventilated_days, pandemic.ventilated_days_lb, pandemic.ventilated_days_ub])
    
    inpatient_daily_cost = DAILY_HOSPITALIZATION_COST[region]["Inpatient"]
    inpatient_daily_cost = inpatient_daily_cost if inpatient_daily_cost is not None else 0
    icu_daily_cost = DAILY_HOSPITALIZATION_COST[region]["ICU bed"]
    icu_daily_cost = icu_daily_cost if icu_daily_cost is not None else 0
    ventilated_daily_cost = DAILY_HOSPITALIZATION_COST[region]["Ventilated ICU bed"]
    ventilated_daily_cost = ventilated_daily_cost if ventilated_daily_cost is not None else 0

    return icu_daily_cost*total_icu_days + ventilated_daily_cost*total_ventilated_days + inpatient_daily_cost*total_hospitalized_days
