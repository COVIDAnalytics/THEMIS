# Authors: Michael L. Li (mlli@mit.edu), Saksham Soni, Baptiste Rossi


from cost_functions.economic_cost.economic_data.economic_params import TOTAL_GDP,GDP_IMPACT,COVID_SICK_DAYS,TOTAL_WORKING_DAYS,TOTAL_LABOR_FORCE
from dateutil.relativedelta import relativedelta
import pandas as pd
from os.path import isfile
from pathlib import Path


def short_term_gdp_costs(pandemic):
    region = pandemic.region
    gdp_data_path = Path(__file__).parent / f"economic_data/gdp/{pandemic.region}.csv"
    if region in TOTAL_GDP and region in GDP_IMPACT and isfile(gdp_data_path):
        gdp_cost = get_gdp_cost(pandemic)
        return gdp_cost
        
    else:
        raise ValueError("Country Not Implemented")
    
    
def get_gdp_cost(pandemic):
    # this function returns the additional gdp loss that would occur under such policy
    # the return value is a list of numbers e.g. [50000,60000,80000,90000] representing the extra gdp loss due to policy, length should correspond with policy.num_months

    if pandemic.policy.policy_type == "actual":
        gdp_data_path = Path(__file__).parent / f"economic_data/gdp/{pandemic.region}.csv"
        gdp_data = pd.read_csv(gdp_data_path)
        gdp_loss = []
        for i in range(pandemic.policy.num_months):
            date = pandemic.policy.start_date + relativedelta(months = i)
            year = date.year
            month = date.month
            gdp_loss_pct = gdp_data[(gdp_data.year == year) & (gdp_data.month == month)].gdp_loss.values[0]
            monthly_gdp_loss = gdp_loss_pct / 100 * TOTAL_GDP[pandemic.region] / 12
            gdp_loss.append(monthly_gdp_loss)
        gdp_cost = -sum(gdp_loss)
        return gdp_cost
    else:
        # we are in hypothetical regime
        gdp_loss = []
        for i in range(len(pandemic.policy.policy_vector)):
            this_month_policy = pandemic.policy.policy_vector[i]
            impact_of_policy = GDP_IMPACT[pandemic.region][this_month_policy]
            monthly_gdp_loss = impact_of_policy / 100 * TOTAL_GDP[pandemic.region] / 12
            gdp_loss.append(monthly_gdp_loss)
        gdp_cost = -sum(gdp_loss)
        sick_worker_cost = pandemic.num_cases / TOTAL_LABOR_FORCE[pandemic.region] * COVID_SICK_DAYS[pandemic.region] / TOTAL_WORKING_DAYS[pandemic.region] * TOTAL_GDP[pandemic.region]
        return gdp_cost + sick_worker_cost