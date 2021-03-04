# Authors: Michael L. Li (mlli@mit.edu),


from .economic_data.economic_params import TOTAL_GDP,GDP_IMPACT
from dateutil.relativedelta import relativedelta
import pandas as pd
from os.path import isfile
from pathlib import Path


def short_term_gdp_costs(policy):
    region = policy.region
    gdp_data_path = Path(__file__).parent / f"economic_data/gdp/{policy.region}.csv"
    if region in TOTAL_GDP and region in GDP_IMPACT and isfile(gdp_data_path):
        gdp_loss = get_gdp_loss(policy)
        return -sum(gdp_loss)
        
    else:
        raise ValueError("Country Not Implemented")
    
    
def get_gdp_loss(policy):
    # this function returns the additional gdp loss that would occur under such policy
    # the return value is a list of numbers e.g. [50000,60000,80000,90000] representing the extra gdp loss due to policy, length should correspond with policy.num_months
    # TODO: This does not yet count the GDP cost of sick workers

    if policy.policy_type == "actual":
        gdp_data_path = Path(__file__).parent / f"economic_data/gdp/{policy.region}.csv"
        gdp_data = pd.read_csv(gdp_data_path)
        gdp_loss = []
        for i in range(policy.num_months):
            date = policy.start_date + relativedelta(months = i)
            year = date.year
            month = date.month
            gdp_loss_pct = gdp_data[(gdp_data.year == year) & (gdp_data.month == month)].gdp_loss.values[0]
            monthly_gdp_loss = gdp_loss_pct / 100 * TOTAL_GDP[policy.region] / 12
            gdp_loss.append(monthly_gdp_loss)
        return gdp_loss
    else:
        # we are in hypothetical regime
        gdp_loss = []
        for i in range(len(policy.policy_vec)):
            this_month_policy = policy.policy_vec[i]
            impact_of_policy = GDP_IMPACT[policy.region][this_month_policy]
            monthly_gdp_loss = impact_of_policy / 100 * TOTAL_GDP[policy.region] / 12
            gdp_loss.append(monthly_gdp_loss)    
        return gdp_loss