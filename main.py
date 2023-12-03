from pandemic_functions.pandemic_cost import PandemicCost
from policy_functions.policy import Policy
from pandemic_functions.pandemic import Pandemic_Factory
import locale
locale.setlocale(locale.LC_ALL, 'en_US')
import sys
import itertools
import pandas as pd
from datetime import datetime
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    '--region', '-r', type=str, required=True,
    help="region code for the region the simulation will be run"
)
parser.add_argument(
    '--startdate', '-sd', type=str, required=False, default="2020-03-01",
    help="start date for the simulation"
)
parser.add_argument(
    '--length', '-l', type=int, required=False, default =3,
    help="number of months the simulation will be run for starting from the start date"
)
arguments = parser.parse_args()
region = arguments.region
start_date = arguments.startdate
policy_length = arguments.length

pandemic_simulator = Pandemic_Factory()


def simulate_actual():
    # create a policy that uses the current policy for Germany, length 3 months starting in 2020-03-01, 
    policy = Policy(policy_type = "actual", start_date = start_date, policy_length = policy_length)
    # simulate the pandemic using such policy
    pandemic = pandemic_simulator.compute_delphi(policy,region=region)
    cost_of_pandemic = PandemicCost(pandemic)
    return cost_of_pandemic

# Should have 5 items in the dict, each representing costs. Economic costs and death costs should be on the order of billions to tens of billions at least. 

def simulate_policy(policy_vect):
    # create a policy that a hypothetical policy for Germany, length 3 months starting in 2020-03-01, 
    policy2 = Policy(policy_type = "hypothetical", start_date = start_date, policy_vector = policy_vect)
    # simulate the pandemic using such policy
    pandemic2 = pandemic_simulator.compute_delphi(policy2,region=region)
    cost_of_pandemic2 = PandemicCost(pandemic2)
    return cost_of_pandemic2

d_scenarii_simulations = {}
d_scenarii_simulations['actual'] = simulate_actual().__dict__

future_policies = [
    'No_Measure', 
    'Restrict_Mass_Gatherings', 
#    'Mass_Gatherings_Authorized_But_Others_Restricted',
   'Restrict_Mass_Gatherings_and_Schools', 
   'Authorize_Schools_but_Restrict_Mass_Gatherings_and_Others',
   'Restrict_Mass_Gatherings_and_Schools_and_Others', 
    'Lockdown'
]

scenarios = [list(t) for t in itertools.product(future_policies, repeat=policy_length)]

for scenario in scenarios:
    print("testing - " + str(scenario) + " ", end='')
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
output_df["country"] = region
output_df["start_date"] = start_date
output_df["policy_length"] = policy_length

output_df = output_df[["country","start_date","policy_length","policy",
        "st_economic_costs", "st_economic_costs_lb", "st_economic_costs_ub", "lt_economic_costs", 
        # "st_term_unemployment_costs", "st_term_gdp_impact", "st_term_sick_worker_cost", "st_term_sick_worker_cost_lb", "st_term_sick_worker_cost_ub",
        "d_costs", "d_costs_lb", "d_costs_ub", "h_costs", "mh_costs", "mh_costs_lb", "mh_costs_ub", 
        "num_cases", "num_cases_lb", "num_cases_ub", "num_deaths", "num_deaths_lb", "num_deaths_ub", 
        "hospitalization_days","icu_days","ventilated_days"]]

time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_df.to_csv('simulation_results/test_result_' + region + '_' + time_stamp + '.csv')
