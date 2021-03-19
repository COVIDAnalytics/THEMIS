from pandemic_cost import get_pandemic_cost
from policy_functions.policy import Policy
from pandemic_functions.pandemic import Pandemic_Factory
import locale
locale.setlocale(locale.LC_ALL, 'en_US')

pandemic_simulator = Pandemic_Factory()
import pandas as pd

def simulate_actual():
    # create a policy that uses the current policy for Germany, length 3 months starting in 2020-03-01, 
    policy = Policy(policy_type = "actual", start_date = "2020-03-01", policy_length = 3)
    # simulate the pandemic using such policy
    pandemic = pandemic_simulator.compute_delphi(policy,region="GM")
    cost_of_pandemic = get_pandemic_cost(pandemic)
    return cost_of_pandemic

# Should have 5 items in the dict, each representing costs. Economic costs and death costs should be on the order of billions to tens of billions at least. 

def simulate_policy(policy_vect):
    # create a policy that a hypothetical policy for Germany, length 3 months starting in 2020-03-01, 
    policy2 = Policy(policy_type = "hypothetical", start_date = "2020-03-01", policy_vector = policy_vect)
    # simulate the pandemic using such policy
    pandemic2 = pandemic_simulator.compute_delphi(policy2,region="GM")
    cost_of_pandemic2 = get_pandemic_cost(pandemic2)
    return cost_of_pandemic2

d_scenarii_simulations = {}
d_scenarii_simulations['actual'] = simulate_actual().__dict__
for scenario in [["No_Measure","No_Measure","No_Measure"],
                 ["Lockdown","Lockdown","Lockdown"]
                 ]:
    d_scenarii_simulations['-'.join(scenario)] = simulate_policy(scenario).__dict__
print(d_scenarii_simulations)
pd.DataFrame.from_dict(d_scenarii_simulations).to_csv('test_result.csv')
#   for k in cost_of_pandemic.__dict__:
#       print(k, " ", locale.format_string("%d", cost_of_pandemic.__dict__[k], grouping=True))