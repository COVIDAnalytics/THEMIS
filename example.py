from pandemic_cost import PandemicCost
from policy_functions.policy import Policy
from pandemic_functions.pandemic import Pandemic_Factory
import locale
locale.setlocale(locale.LC_ALL, 'en_US')
import sys

pandemic_simulator = Pandemic_Factory()
import pandas as pd

country = "GM"
start_date = "2020-03-01"
policy_length = 3

def simulate_actual():
    # create a policy that uses the current policy for Germany, length 3 months starting in 2020-03-01, 
    policy = Policy(policy_type = "actual", start_date = start_date, policy_length = policy_length)
    # simulate the pandemic using such policy
    pandemic = pandemic_simulator.compute_delphi(policy,region=country)
    cost_of_pandemic = PandemicCost(pandemic)
    return cost_of_pandemic

# Should have 5 items in the dict, each representing costs. Economic costs and death costs should be on the order of billions to tens of billions at least. 

def simulate_policy(policy_vect):
    # create a policy that a hypothetical policy for Germany, length 3 months starting in 2020-03-01, 
    policy2 = Policy(policy_type = "hypothetical", start_date = start_date, policy_vector = policy_vect)
    # simulate the pandemic using such policy
    pandemic2 = pandemic_simulator.compute_delphi(policy2,region=country)
    cost_of_pandemic2 = PandemicCost(pandemic2)
    return cost_of_pandemic2

d_scenarii_simulations = {}
d_scenarii_simulations['actual'] = simulate_actual().__dict__

future_policies = [
    'No_Measure', 'Restrict_Mass_Gatherings', 'Mass_Gatherings_Authorized_But_Others_Restricted',
    'Restrict_Mass_Gatherings_and_Schools', 'Authorize_Schools_but_Restrict_Mass_Gatherings_and_Others',
    'Restrict_Mass_Gatherings_and_Schools_and_Others', 'Lockdown'
]

for policies in future_policies:
    scenario = [policies] * policy_length
    print("testing - " + policies + " ", end='')
    try:
        d_scenarii_simulations['-'.join(scenario)] = simulate_policy(scenario).__dict__
        print("OK")
    except :
        print("Unexpected error:", sys.exc_info())
        print("NOK")

output_df = pd.DataFrame.from_dict(d_scenarii_simulations, orient="index")
del output_df["policy"]
output_df.reset_index(inplace=True)
output_df = output_df.rename(columns = {'index':'policy'})
output_df["country"] = country
output_df["start_date"] = start_date
output_df["policy_length"] = policy_length

output_df = output_df[["country","start_date","policy_length","policy","st_economic_costs","lt_economic_costs",
                       "d_costs","h_costs","mh_costs","num_cases","num_deaths","hospitalization_days","icu_days","ventilated_days"]]

output_df.to_csv('simulation_results/test_result.csv')
#   for k in cost_of_pandemic.__dict__:
#       print(k, " ", locale.format_string("%d", cost_of_pandemic.__dict__[k], grouping=True))