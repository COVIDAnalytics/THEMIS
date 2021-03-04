# Authors: Michael L. Li (mlli@mit.edu),



from cost_functions.economic_cost.economic_data.economic_params import UNEMPLOYMENT_COST,EMPLOYMENT_IMPACT,TOTAL_LABOR_FORCE
from dateutil.relativedelta import relativedelta
import pandas as pd
from os.path import isfile
from pathlib import Path



def short_term_unemployment_costs(pandemic):
    region = pandemic.region
    unemployment_data_path = Path(__file__).parent / f"economic_data/unemployment/{pandemic.region}.csv"
    if region in UNEMPLOYMENT_COST and region in EMPLOYMENT_IMPACT and region in TOTAL_LABOR_FORCE and isfile(unemployment_data_path):
        employment_loss = get_employment_loss(pandemic)
        pp_cost_year = UNEMPLOYMENT_COST[region]
        return pp_cost_year * sum(employment_loss) * 1 / 12
        
    else:
        raise ValueError("Country Not Implemented")
    
    
def get_employment_loss(pandemic):
    # this function returns the additional unemployed people per month that would occur under such policy
    # the return value is a list of numbers e.g. [50000,60000,80000,90000] representing the extra unemployment due to policy, length should correspond with policy_length or policy_vec

    if pandemic.policy.policy_type == "actual":
        unemployment_data_path = Path(__file__).parent / f"economic_data/unemployment/{pandemic.region}.csv"
        unemployment_data = pd.read_csv(unemployment_data_path)
        total_gain = []
        for i in range(pandemic.policy.num_months):
            date = pandemic.policy.start_date + relativedelta(months = i)
            year = date.year
            month = date.month
            unemployment_gain_pct = unemployment_data[(unemployment_data.year == year) & (unemployment_data.month == month)].unemployment_gain.values[0]
            monthly_gain = unemployment_gain_pct / 100 * TOTAL_LABOR_FORCE[pandemic.region]
            total_gain.append(monthly_gain)
        return total_gain
    else:
        # we are in hypothetical regime
        total_gain = []
        for i in range(len(pandemic.policy.policy_vector)):
            this_month_policy = pandemic.policy.policy_vector[i]
            impact_of_policy = EMPLOYMENT_IMPACT[pandemic.region][this_month_policy]
            monthly_gain = impact_of_policy / 100 * TOTAL_LABOR_FORCE[pandemic.region]
            total_gain.append(monthly_gain)    
        return total_gain