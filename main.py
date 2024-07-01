from pandemic_functions.pandemic_cost import PandemicCost
from policy_functions.policy import Policy
from pandemic_functions.pandemic import Pandemic_Factory
import locale
locale.setlocale(locale.LC_ALL, 'en_US')
import sys
import os
import multiprocessing as mp
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
parser.add_argument(
    '--samplegammas', '-g', type=int, required=False, default =0,
    help="if gamma values should be sampled from a distribution for error margin calculation"
)

arguments = parser.parse_args()
region = arguments.region
start_date = arguments.startdate
policy_length = arguments.length
sample_gammas = bool(arguments.samplegammas)
n_proc = os.cpu_count() // 2

def simulate_actual(pandemic_simulator):
    # create a policy that uses the current policy for Germany, length 3 months starting in 2020-03-01, 
    policy = Policy(policy_type = "actual", start_date = start_date, policy_length = policy_length)
    # simulate the pandemic using such policy
    pandemic = pandemic_simulator.compute_delphi(policy,region=region)
    cost_of_pandemic = PandemicCost(pandemic)
    return "actual", cost_of_pandemic.__dict__

# Should have 5 items in the dict, each representing costs. Economic costs and death costs should be on the order of billions to tens of billions at least. 

def simulate_policy(args):
    pandemic_simulator, policy_vect = args
    # create a policy that a hypothetical policy for Germany, length 3 months starting in 2020-03-01, 
    policy2 = Policy(policy_type = "hypothetical", start_date = start_date, policy_vector = policy_vect)
    # simulate the pandemic using such policy
    pandemic2 = pandemic_simulator.compute_delphi(policy2,region=region,sample_gammas=sample_gammas)
    cost_of_pandemic2 = PandemicCost(pandemic2)
    return '-'.join(policy_vect), cost_of_pandemic2.__dict__

if __name__ == "__main__":
    pandemic_simulator = Pandemic_Factory()
    scenarii_simulations = []
    scenarii_simulations.append(simulate_actual(pandemic_simulator))

    future_policies = [
        'No_Measure', 
        'Restrict_Mass_Gatherings', 
    #    'Mass_Gatherings_Authorized_But_Others_Restricted',
    'Restrict_Mass_Gatherings_and_Schools', 
    'Authorize_Schools_but_Restrict_Mass_Gatherings_and_Others',
    'Restrict_Mass_Gatherings_and_Schools_and_Others', 
        'Lockdown'
    ]

    scenarios = [(pandemic_simulator, list(t)) for t in itertools.product(future_policies, repeat=policy_length)]

    if n_proc == 1:
        for scenario in scenarios:
            print("Running - " + str(scenario[1]) + " ", end='')
            try:
                scenarii_simulations.append(simulate_policy(scenario))
                print("OK")
            except :
                print("Unexpected error:", sys.exc_info())
                print("NOK")
    else:
        with mp.Pool(processes=n_proc) as pool:
            results = pool.map(simulate_policy, scenarios)
        scenarii_simulations.extend(results)

    d_scenarii_simulations = dict(scenarii_simulations)
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
            "d_costs", "d_costs_lb", "d_costs_ub", "h_costs", "h_costs_lb", "h_costs_ub", "mh_costs", "mh_costs_lb", "mh_costs_ub", 
            "num_cases", "num_cases_lb", "num_cases_ub", "num_deaths", "num_deaths_lb", "num_deaths_ub", 
            "hospitalization_days", "hospitalization_days_lb", "hospitalization_days_ub", 
            "icu_days", "icu_days_lb", "icu_days_ub", "ventilated_days", "ventilated_days_lb", "ventilated_days_ub"]]

    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_df.to_csv('simulation_results/test_result_' + region + '_' + time_stamp + '.csv')
