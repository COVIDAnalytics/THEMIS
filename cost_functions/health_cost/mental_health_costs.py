## Authors: Baptiste
from cost_functions.health_cost.health_data.health_params import MENTAL_HEALTH_COST
from pandemic_functions.delphi_functions.DELPHI_model_policy_scenarios import read_policy_data_us_only, read_oxford_country_policy_data
from pandemic_functions.pandemic_params import region_symbol_country_dict
def mental_health_costs(pandemic):
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

    cumulated_sick = pandemic.num_cases
    depressed_patients = MH_DATA["exposed_health_workers"] * MH_DATA["depression_rate_hworkers_normal"]
    depressed_patients +=  cumulated_sick * MH_DATA["depression_rate_sick"]
    gen_pop_depression = MH_DATA["gen_population_over14"] * MH_DATA["depression_gen_pop"] * lockdown_months/12.
    depressed_patients += gen_pop_depression

    ptsd_patients = MH_DATA["exposed_health_workers"] * MH_DATA["ptsd_rate_hworkers"]
    ptsd_patients += cumulated_sick * MH_DATA["ptsd_rate_sick"]

    return depressed_patients * MH_DATA["depression_cost"] + ptsd_patients * MH_DATA["ptsd_cost"]
