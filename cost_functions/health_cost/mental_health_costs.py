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
    if pandemic.policy.policy_type == "hypothetical":
        lockdown_months = 0
        for lockdown_policy in MH_DATA["lockdown_equivalent_policies"]:
            lockdown_months +=  sum(map(lambda a: a == lockdown_policy, pandemic.policy.policy_vector))
    else:
        country, province = region_symbol_country_dict[region]
        if country == 'US':
            policy_data = read_policy_data_us_only(state=province, start_date=pandemic.policy.start_date, end_date=pandemic.policy.end_date)
        else:
            policy_data = read_oxford_country_policy_data(country=country, start_date=pandemic.policy.start_date, end_date=pandemic.policy.end_date)
        total_lockdown_days = 0
        for lockdown_policy in MH_DATA["lockdown_equivalent_policies"]:
            total_lockdown_days += sum(policy_data[lockdown_policy])
        lockdown_months = total_lockdown_days / 30

    cumulated_sick = np.array([pandemic.num_cases, pandemic.num_cases_lb, pandemic.num_cases_ub])
    # depressed_patients = MH_DATA["exposed_health_workers"] * MH_DATA["depression_rate_hworkers_normal"] * lockdown_months/12.
    depressed_patients =  cumulated_sick * max(MH_DATA["depression_rate_inc_sick"] * 14/365, MH_DATA["depression_rate_inc_gen_population"]* lockdown_months/12.)
    gen_pop_depression = (MH_DATA["gen_population_over14"] - cumulated_sick )* MH_DATA["depression_rate_inc_gen_population"] * lockdown_months/12.
    depressed_patients += gen_pop_depression

    ptsd_patients = MH_DATA["exposed_health_workers"] * MH_DATA["ptsd_rate_inc_hworkers"]
    ptsd_patients += cumulated_sick * MH_DATA["ptsd_rate_inc_sick"]

    return tuple(depressed_patients * MH_DATA["depression_cost"] + ptsd_patients * MH_DATA["ptsd_cost"])
