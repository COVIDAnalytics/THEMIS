## Authors: Baptiste
from cost_functions.health_cost.health_data.health_params import MENTAL_HEALTH_COST

def mental_health_costs(pandemic):
    region = pandemic.region
    MH_DATA = MENTAL_HEALTH_COST[region]
    if pandemic.policy.policy_type == "hypothetical":
        lockdown_months  = sum(map(lambda a: a == 'Lockdown', pandemic.policy.policy_vector))
    else:
        lockdown_months = MH_DATA["lockdown_months"]

    cumulated_sick = pandemic.num_cases
    depressed_patients = MH_DATA["exposed_health_workers"] * MH_DATA["depression_rate_hworkers_normal"]
    depressed_patients +=  cumulated_sick * MH_DATA["depression_rate_sick"]
    gen_pop_depression = MH_DATA["gen_population_over14"] * MH_DATA["depression_gen_pop"] * lockdown_months/12.
    depressed_patients += gen_pop_depression

    ptsd_patients = MH_DATA["exposed_health_workers"] * MH_DATA["ptsd_rate_hworkers"]
    ptsd_patients += cumulated_sick * MH_DATA["ptsd_rate_sick"]

    return depressed_patients * MH_DATA["depression_cost"] + ptsd_patients * MH_DATA["ptsd_cost"]
