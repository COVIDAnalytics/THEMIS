from pandemic_cost import PandemicCost
from policy_functions.policy import Policy
from pandemic_functions.pandemic import Pandemic_Factory
import locale
locale.setlocale(locale.LC_ALL, 'en_US')
import sys

pandemic_simulator = Pandemic_Factory()
import pandas as pd

def simulate_actual():
    # create a policy that uses the current policy for Germany, length 3 months starting in 2020-03-01, 
    policy = Policy(policy_type = "actual", start_date = "2020-03-01", policy_length = 3)
    # simulate the pandemic using such policy
    pandemic = pandemic_simulator.compute_delphi(policy,region="GM")
    cost_of_pandemic = PandemicCost(pandemic)
    return cost_of_pandemic

# Should have 5 items in the dict, each representing costs. Economic costs and death costs should be on the order of billions to tens of billions at least. 

def simulate_policy(policy_vect):
    # create a policy that a hypothetical policy for Germany, length 3 months starting in 2020-03-01, 
    policy2 = Policy(policy_type = "hypothetical", start_date = "2020-03-01", policy_vector = policy_vect)
    # simulate the pandemic using such policy
    pandemic2 = pandemic_simulator.compute_delphi(policy2,region="GM")
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
    scenario = [policies,policies,policies]
    print("testing - " + policies + " ", end='')
    try:
        d_scenarii_simulations['-'.join(scenario)] = simulate_policy(scenario).__dict__
        print("OK")
    except :
        print("Unexpected error:", sys.exc_info())
        print("NOK")

pd.DataFrame.from_dict(d_scenarii_simulations, orient="index").to_csv('simulation_results/test_result.csv')
#   for k in cost_of_pandemic.__dict__:
#       print(k, " ", locale.format_string("%d", cost_of_pandemic.__dict__[k], grouping=True))