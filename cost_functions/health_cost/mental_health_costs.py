## Authors: Baptiste, Michael L. Li, Saksham Soni
from cost_functions.health_cost.health_data.health_params import MENTAL_HEALTH_COST
from pandemic_functions.delphi_functions.DELPHI_model_policy_scenarios import read_policy_data_us_only, read_oxford_country_policy_data
from pandemic_functions.pandemic_params import region_symbol_country_dict
import numpy as np

def mental_health_costs(pandemic):
    """
    Function that returns the cost of impact on mental health of the population
    Parameters:
        - pandemic: Pandemic object containing the information of the region and duration that is being analyzed
    Returns:
        - tuple, (value, lower bound, upper bound) cost of impact on mental health in local currency
    """
    region = pandemic.region
    MH_DATA = MENTAL_HEALTH_COST[region]
    gamma = pandemic.dict_region_policy_gamma
    gamma_min = min(gamma.values())

    if pandemic.policy.policy_type == "hypothetical":
        k = len(pandemic.policy.policy_vector)
        adjust_factor = sum(
            (1 - gamma[pandemic.policy.policy_vector[l]]) / (1 - gamma_min)
            for l in range(k)
        ) / 12.0
    else:
        adjust_factor = sum(
            pandemic.dict_region_policy_counts[x] * (1 - gamma[x])
            for x in pandemic.dict_region_policy_counts
        ) / (365.0 * (1 - gamma_min))

    cumulated_sick = np.array([pandemic.num_cases, pandemic.num_cases_lb, pandemic.num_cases_ub])

    depressed_patients = MH_DATA["gen_population_over14"] * MH_DATA["depression_rate_inc_gen_population"] * adjust_factor

    ptsd_patients = MH_DATA["exposed_health_workers"] * MH_DATA["ptsd_rate_inc_hworkers"]
    ptsd_patients += cumulated_sick * MH_DATA["ptsd_rate_inc_sick"]

    return tuple(depressed_patients * MH_DATA["depression_cost"] + ptsd_patients * MH_DATA["ptsd_cost"])
